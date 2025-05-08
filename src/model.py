import torch
print(f"torch version = <{torch.__version__}>")
import torch_geometric
print(f"torch geometric version = <{torch_geometric.__version__}>")
from typing import Optional
from typing import Union, Tuple
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch_geometric.utils import softmax
from torch_sparse import SparseTensor
import torch.nn as nn
import torch.nn.functional as F

"""
GRAPH RANK MODEL
"""

class GraFrankConv(MessagePassing):
    """
    Modality-specific neighbor aggregation in GraFrank implemented by stacking message-passing layers that are
    parameterized by friendship attentions over individual node features and pairwise link features.
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):
        ## Initialize all the linear layers and attention parameters
        kwargs.setdefault('aggr', 'add')
        super(GraFrankConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.negative_slope = 0.2
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.self_linear = nn.Linear(in_channels[1], out_channels, bias=bias)
        self.message_linear = nn.Linear(in_channels[0], out_channels, bias=bias)

        self.attn = nn.Linear(out_channels, 1, bias=bias)
        self.attn_i = nn.Linear(out_channels, 1, bias=bias)

        self.lin_l = nn.Linear(out_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(out_channels, out_channels, bias=False)

        self.reset_parameters()
        self.dropout = 0

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        x_l, x_r = x[0], x[1]
        self_emb = self.self_linear(x_r)
        alpha_i = self.attn_i(self_emb)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha_i,
                             edge_attr=edge_attr, size=size)
        out = self.lin_l(out) + self.lin_r(self_emb)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, alpha_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        message = torch.cat([x_j, edge_attr], dim=-1)
        out = self.message_linear(message)
        alpha = self.attn(out) + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i) ### normalize attention weights
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = out * alpha                          ### weighted sum of messages
        return out

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




class CrossModalityAttention(nn.Module):
    """
    Cross-Modality Fusion in GraFrank implemented by an attention mechanism
    across the K modalities.
    """

    def __init__(self, hidden_channels):
        super(CrossModalityAttention, self).__init__()
        self.hidden_channels = hidden_channels
        self.multi_linear = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.multi_attn = nn.Sequential(self.multi_linear, nn.Tanh(),
                                        nn.Linear(hidden_channels, 1, bias=True))

    def forward(self, modality_x_list):
        """
        :param modality_x_list: list of modality-specific node embeddings.
        :return: final node embedding after fusion.
        """
        result = torch.cat([x.relu().unsqueeze(-2) for x in modality_x_list], -2)  # [...., K, hidden_channels]
        wts = torch.softmax(self.multi_attn(result).squeeze(-1), dim=-1)
        return torch.sum(wts.unsqueeze(-1) * self.multi_linear(result), dim=-2)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.hidden_channels,
                                   self.hidden_channels)
    

class GraFrank(nn.Module):
    """
    GraFrank Model for Multi-Faceted Friend Ranking with multi-modal node features and pairwise link features.
    (a) Modality-specific neighbor aggregation: modality_convs
    (b) Cross-modality fusion layer: cross_modality_attention
    """

    def __init__(self, in_channels, hidden_channels, edge_channels, num_layers, input_dim_list):
        """
        :param in_channels: total cardinality of node features.
        :param hidden_channels: latent embedding dimensionality.
        :param edge_channels: number of link features.
        :param num_layers: number of message passing layers.
        :param input_dim_list: list containing the cardinality of node features per modality.
        """
        super(GraFrank, self).__init__()
        self.num_layers = num_layers
        self.modality_convs = nn.ModuleList()
        self.edge_channels = edge_channels
        # we assume that the input features are first partitioned and then concatenated across the K modalities.
        self.input_dim_list = input_dim_list

        # for loop to stack the convolutional layers
        for inp_dim in self.input_dim_list:
            modality_conv_list = nn.ModuleList()
            for i in range(num_layers):
                in_channels = in_channels if i == 0 else hidden_channels
                modality_conv_list.append(GraFrankConv((inp_dim + edge_channels,
                                                        inp_dim), hidden_channels))

            self.modality_convs.append(modality_conv_list)

        self.cross_modality_attention = CrossModalityAttention(hidden_channels)

    def forward(self, x, adjs, edge_attrs):
        """ Compute node embeddings by recursive message passing, followed by cross-modality fusion.
        :param x: node features [B', in_channels] where B' is the number of nodes (and neighbors) in the mini-batch.
        :param adjs: list of sampled edge indices per layer (EdgeIndex format in PyTorch Geometric) in the mini-batch.
        :param edge_attrs: [E', edge_channels] where E' is the number of sampled edge indices per layer in the mini-batch.
        :return: node embeddings. [B, hidden_channels] where B is the number of target nodes in the mini-batch.
        """
        result = []   ### Consider each modality separately
        for k, convs_k in enumerate(self.modality_convs):
            emb_k = None
            for i, ((edge_index, _, size), edge_attr) in enumerate(zip(adjs, edge_attrs)):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x_list = torch.split(x, split_size_or_sections=self.input_dim_list, dim=-1)  # modality partition
                x_target_list = torch.split(x_target, split_size_or_sections=self.input_dim_list, dim=-1)
                x_k, x_target_k = x_list[k], x_target_list[k]

                emb_k = convs_k[i]((x_k, x_target_k), edge_index, edge_attr=edge_attr)

                if i != self.num_layers - 1:
                    emb_k = emb_k.relu()
                    emb_k = F.dropout(emb_k, p=0.5, training=self.training)

            result.append(emb_k)
        return self.cross_modality_attention(result)

    def full_forward(self, x, edge_index, edge_attr):
        """ Auxiliary function to compute node embeddings for all nodes at once for small graphs.
        :param x: node features [N, in_channels] where N is the total number of nodes in the graph.
        :param edge_index: edge indices [2, E] where E is the total number of edges in the graph.
        :param edge_attr: link features [E, edge_channels] across all edges in the graph.
        :return: node embeddings. [N, hidden_channels] for all nodes in the graph.
        """
        x_list = torch.split(x, split_size_or_sections=self.input_dim_list, dim=-1)  # modality partition
        result = []
        for k, convs_k in enumerate(self.modality_convs):
            x_k = x_list[k]
            emb_k = None
            for i, conv in enumerate(convs_k):
                emb_k = conv(x_k, edge_index, edge_attr=edge_attr)

                if i != self.num_layers - 1:
                    emb_k = emb_k.relu()
                    emb_k = F.dropout(emb_k, p=0.5, training=self.training)

            result.append(emb_k)
        return self.cross_modality_attention(result)
 