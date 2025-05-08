import torch
print(f"torch version = <{torch.__version__}>")
import torch_geometric
print(f"torch geometric version = <{torch_geometric.__version__}>")
import numpy as np
import pandas as pd
import warnings
from copy import deepcopy
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import train_test_split_edges
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from src.dataset import *
from src.model import *


"""
Load the dataset
"""
print("Loading network...")
feat_file_name = "Data/feature_map.txt"
dataset_folder_path = "Data"
load_network(dataset_folder_path)
print("Loading done.")


"""
Declare the model and training parameters
"""
device = torch.device('cpu')
warnings.filterwarnings("ignore") # we dont want to hear PyG scream

# Convert networkx graph to PyG; convert data to cpu
myGraph = deepcopy(network).to_directed()
data = from_networkx(myGraph)
data.to('cpu')

#splitting the dataset
data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
data.to("cpu")

## input dim list assumes that the node features are first
## 1283 is the number of node features
## since we don't really have modalities in this dataset, and
## GraFrank pays special attention to them, we create pseudo modalities
## by partitioning the edge features into 4 different sets =P
model = GraFrank(1283, hidden_channels=64, edge_channels=5, num_layers=4,
                 input_dim_list=[350, 350, 350, 233])
model = model.to("cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




def get_link_labels(pos_edge_index, neg_edge_index, devce):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    # used as our ground truth
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=devce)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train_model():
    model.train()

    # sampling negative edges
    """
    if we only train on positive edges (supervision edges) then our GNN wouldvery quickly learn to output
    all positive predictions. Thus we sample non-existent edges (in any of the splits) and label them as
    negative (false) edges and use them to train the GNN
    """
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, # positive edges
        num_nodes=data.num_nodes, # number of nodes
        num_neg_samples=data.train_pos_edge_index.size(1)).to(device) # number of neg_sample equal to number of pos_edges

    optimizer.zero_grad()

    data.train_pos_edge_index.to(device)

    # adding the adjacency representations of both edge types together
    edge_index = torch.cat([data.train_pos_edge_index, neg_edge_index], dim=-1)

    x = data.node_feature
    # x = torch.stack(data.node_feature)
    x.to(device)
    # edge_feature = torch.stack(data.edge_feature)
    edge_feature = data.edge_feature
    edge_feature.to(device)

    # creating embeddings for all nodes
    out = model.full_forward(x, data.train_pos_edge_index, edge_feature[:data.train_pos_edge_index.shape[1]])

    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index, "cpu")

    # retaining only the dot products for the nodes that span our training edges
    logits = (out[edge_index[0]] * out[edge_index[1]]).sum(dim=-1)

    # training with simple binary cross entropy
    loss = BCEWithLogitsLoss()(logits, link_labels)
    loss.backward()
    optimizer.step()

    train_acc = roc_auc_score(link_labels.cpu(), logits.sigmoid().flatten().cpu().detach().numpy())

    # torch.cpu.empty_cache()
    return loss, train_acc
    

def test_model(output=False):
    # defining the test function
    # very similar to the train function but the sampling was already done for us
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']

        # x = torch.stack(data.node_feature)
        x = data.node_feature
        x.to("cpu")
        # edge_feature = torch.stack(data.edge_feature)
        edge_feature = data.edge_feature
        edge_feature.to("cpu")

        out = model.full_forward(x, pos_edge_index, edge_feature[:pos_edge_index.shape[1]])
        link_labels = get_link_labels(pos_edge_index, neg_edge_index, "cpu")

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

        logits = (out[edge_index[0]] * out[edge_index[1]]).sum(dim=-1)

        loss = BCEWithLogitsLoss()(logits, link_labels)
        perfs.append(loss)
        link_probs = logits.sigmoid() # apply sigmoid        
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.flatten().cpu().detach().numpy())) #compute roc_auc score
        if prefix=="test" and output:
          return perfs, link_labels.cpu(), link_probs.flatten().cpu().detach().numpy()
    return perfs


print("Training the model...")
losses = []
train_accs = []
val_accs = []
test_accs = []
val_losses = []
test_losses = []

print_freq = 10 # print every 10 epochs
num_epochs = 80
for epoch in range(num_epochs+1):
    loss, train_acc = train_model()
    # torch.cpu.empty_cache()
    losses.append(loss)
    train_accs.append(train_acc)
    if epoch % print_freq == 0:
      print(f'Epoch: {epoch:03d}')
      print(f'Train Loss:\t{loss:.4f}\t\tTrain ROC_AUC:\t{train_acc:.4f}')
      test_acc = test_model()
      val_accs.append(test_acc[1])
      test_accs.append(test_acc[3])
      val_losses.append(test_acc[0])
      test_losses.append(test_acc[2])
      print(f'Val Loss:\t{val_losses[epoch//print_freq-1]:.4f}\t\tVal ROC_AUC:\t{val_accs[epoch//print_freq-1]:.4f}')
      print(f'Test Loss:\t{test_losses[epoch//print_freq-1]:.4f}\t\tTest ROC_AUC:\t{test_accs[epoch//print_freq-1]:.4f} ')
      print("---------------------------------------------")



print("Evaluation...")
def generate_random_important_statistics():
  metrics, truths, preds = test_model(True)
  preds = np.rint(preds) # round to nearest integer so sklearn metrics can be used
  print("Test Accuracy:\t", accuracy_score(truths, preds))
  print("Test Precision:\t", precision_score(truths, preds))
  print("Test Recall:\t", recall_score(truths, preds))
  print("Test F1 Score:\t", f1_score(truths, preds))

generate_random_important_statistics()




print("Saving predictions in <predicted_links.csv>...")
def test_model_and_save_csv(csv_path="predicted_links.csv"):
    model.eval()
    all_perfs = []
    
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']

        x = data.node_feature.to("cpu")
        edge_feature = data.edge_feature.to("cpu")

        out = model.full_forward(
            x, 
            pos_edge_index, 
            edge_feature[:pos_edge_index.shape[1]]
        )
        link_labels = get_link_labels(pos_edge_index, neg_edge_index, "cpu")
        
        # Stack positive then negative edges for joint scoring
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (out[edge_index[0]] * out[edge_index[1]]).sum(dim=1)
        probs = logits.sigmoid().cpu().detach().numpy()
        
        # Performance metrics
        loss = BCEWithLogitsLoss()(logits, link_labels)
        auc  = roc_auc_score(link_labels.cpu(), probs)
        all_perfs.extend([loss, auc])
        
        # If this is the test split, save predictions to CSV
        if prefix == "test":
            src = edge_index[0].cpu().numpy()
            tgt = edge_index[1].cpu().numpy()
            lbl = link_labels.cpu().numpy()
            
            df = pd.DataFrame({
                "ID1": src,
                "ID2": tgt,
                "label": lbl,
                "score": probs
            })
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(df)} predicted links to {csv_path}")
    
    return all_perfs

perfs = test_model_and_save_csv("predicted_links.csv")
# perfs will be [val_loss, val_auc, test_loss, test_auc]

