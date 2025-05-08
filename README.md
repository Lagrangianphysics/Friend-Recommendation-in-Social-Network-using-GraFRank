# Friend-Recommendation-in-Social-Network-using-GraFRank

# Facebook Social Circles Dataset

### Source
The dataset is provided by the Stanford Network Analysis Project (SNAP) and can be accessed at [https://snap.stanford.edu/data/ego-Facebook.html](https://snap.stanford.edu/data/ego-Facebook.html).

### Description
This dataset contains anonymized Facebook data collected from survey participants using the Facebook app. It represents the social circles (ego networks) of users, where each ego network consists of a central user (ego) and their friends. The dataset is undirected and unweighted, meaning that all connections are mutual and have no associated weights.

### Data Storage
- **Graph Representation**: The dataset is stored as a collection of ego networks, where each ego network is represented as a graph.
- **Files**:
  - `facebook_combined.txt`: Contains the edges of the combined graph, where each line represents an edge between two nodes (user IDs).
  - `ego/*.edges`: Contains the edges for each ego network, where each file corresponds to a specific ego.
  - `ego/*.feat`: Contains the features of nodes in each ego network. Each row corresponds to a node, and the columns represent binary features.
  - `ego/*.featnames`: Contains the names of the features in the `.feat` files, providing a mapping of feature indices to their descriptions.

### Key Characteristics
- **Nodes**: Represent Facebook users.
- **Edges**: Represent friendships between users.
- **Features**: Binary attributes of users, such as profile information (e.g., education, location, etc.).
- **Ego Networks**: Each ego network is centered around a specific user and includes their friends and the connections between them.


### Note:
- The data has been anonymized to protect user privacy.
- The dataset is suitable for studying the structure and dynamics of social networks.

---
## 5.1 Loading the Dataset

Our data loading code will load the dataset into a single NetworkX graph. We then convert the NetworkX graph to a PyG graph object, which we feed into the GNN.

The process of loading the dataset and converting it into the required data structure for use in the Graph Neural Network (GNN) is a complex and intricate task. To streamline this process and ensure modularity, the complete implementation for dataset loading and conversion has been encapsulated in the `src.dataset` module. 

Here, we simply utilize the functionality provided by `src.dataset` to load the dataset efficiently.
