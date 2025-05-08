import os
import torch
import networkx as nx
import glob

"""
Load the dataset
"""

feature_index = {}  #numeric index to name
inverted_feature_index = {} #name to numeric index
network = nx.Graph()
ego_nodes = []

def parse_featname_line(line):
    line = line[(line.find(' '))+1:]  # chop first field
    split = line.split(';')
    name = ';'.join(split[:-1]) # feature name
    index = int(split[-1].split(" ")[-1]) #feature index
    return index, name

def load_features(dataset_folder_path):
    # may need to build the index first
    feat_file_name = f"{dataset_folder_path}/feature_map.txt"
    if not os.path.exists(feat_file_name):
        feat_index = {}
        # build the index from data/*.featnames files
        featname_files = glob.iglob(f"{dataset_folder_path}/facebook/*.featnames" )
        for featname_file_name in featname_files:
            featname_file = open(featname_file_name, 'r')
            for line in featname_file:
                # example line:
                # 0 birthday;anonymized feature 376
                index, name = parse_featname_line(line)
                feat_index[index] = name
            featname_file.close()
        keys = feat_index.keys()
        keys = sorted(keys)
        out = open(feat_file_name,'w')
        for key in keys:
            out.write("%d %s\n" % (key, feat_index[key]))
        out.close()

    # index built, read it in (even if we just built it by scanning)
    global feature_index
    global inverted_feature_index
    index_file = open(feat_file_name,'r')
    for line in index_file:
        split = line.strip().split(' ')
        key = int(split[0])
        val = split[1]
        feature_index[key] = val
    index_file.close()

    for key in feature_index.keys():
        val = feature_index[key]
        inverted_feature_index[val] = key

def load_nodes(dataset_folder_path):
    assert len(feature_index) > 0, "call load_features() first"
    global network
    global ego_nodes

    # get all the node ids by looking at the files
    ego_nodes = [int(x.split("\\")[-1].split('.')[0]) for x in glob.glob("Data/facebook/*.featnames")]
    node_ids = ego_nodes

    # parse each node
    for ego_node_id in node_ids:
        # not so great
        featname_file = open(f"{dataset_folder_path}/facebook/{ego_node_id}.featnames", 'r')
        feat_file     = open(f"{dataset_folder_path}/facebook/{ego_node_id}.feat", 'r')
        egofeat_file  = open(f"{dataset_folder_path}/facebook/{ego_node_id}.egofeat", 'r')
        edge_file     = open(f"{dataset_folder_path}/facebook/{ego_node_id}.edges", 'r')

        # 0 1 0 0 0 ...
        ego_features = [int(x) for x in egofeat_file.readline().split(' ')]

        # Add ego node if not already contained in network
        if not network.has_node(ego_node_id):
            network.add_node(ego_node_id)
            network.nodes[ego_node_id]['node_feature'] = torch.zeros(len(feature_index))
            network.nodes[ego_node_id]['ego_id'] = ego_node_id

        # parse ego node
        i = 0
        for line in featname_file:
            key, val = parse_featname_line(line)
            # Update feature value if necessary
            if ego_features[i] + 1 > network.nodes[ego_node_id]['node_feature'][key]:
                network.nodes[ego_node_id]['node_feature'][key] = ego_features[i] + 1
            i += 1

        # parse neighboring nodes
        for line in feat_file:
            featname_file.seek(0)
            split = [int(x) for x in line.split(' ')]
            node_id = split[0]
            features = split[1:]

            # Add node if not already contained in network
            if not network.has_node(node_id):
                network.add_node(node_id)
                network.nodes[node_id]['node_feature'] = torch.zeros(len(feature_index))
                network.nodes[node_id]['ego_id'] = ego_node_id

            i = 0
            for line in featname_file:
                key, val = parse_featname_line(line)
                # Update feature value if necessary
                if features[i] + 1 > network.nodes[node_id]['node_feature'][key]:
                    network.nodes[node_id]['node_feature'][key] = features[i] + 1
                i += 1

        featname_file.close()
        feat_file.close()
        egofeat_file.close()
        edge_file.close()

def load_edges(dataset_folder_path):
    # works perfect
    global network
    assert network.order() > 0, "call load_nodes() first"
    edge_file = open(f"{dataset_folder_path}/facebook_combined.txt","r")
    for line in edge_file:
        # nodefrom nodeto
        split = [int(x) for x in line.split(" ")]
        node_from = split[0]
        node_to = split[1]
        network.add_edge(node_from, node_to)

def load_edge_features():
    global network
    # adding dummy edge features since this dataset does not contain any
    # but GraFrank requires edge features
    temp = {edge: torch.tensor([1., 1., 1., 1., 1.]) for edge in network.edges}
    nx.set_edge_attributes(network, temp, "edge_feature")

def load_network(dataset_folder_path):
    # Load the entire network. network is now the nx graph object
    load_features(dataset_folder_path)
    load_nodes(dataset_folder_path)
    load_edges(dataset_folder_path)
    load_edge_features()