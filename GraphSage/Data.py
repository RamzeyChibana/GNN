import os
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler



CORA = "Data\\cora"

def load_cora(): 
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes,num_feats),dtype=np.float32)
    labels = np.empty((num_nodes,1))
    nodes_id = {}
    label_id = {}
    adj_lists = defaultdict(set)
    with open(os.path.join(CORA,"cora.content")) as fn : #nodes features and node id
        for i,line in enumerate(fn):
            features = line.strip().split()
            feat_data[i,:]=features[1:-1]
            nodes_id[features[0]] = i 
            if features[-1] not in label_id :
                label_id[features[-1]] = len(label_id)
            labels[i] = label_id[features[-1]]
    
    with open(os.path.join(CORA,"cora.cites")) as fe : #nodes conections (edges)
        for line in fe :
            edge = line.strip().split()
            paper1 = nodes_id[edge[0]]
            paper2 = nodes_id[edge[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    adj_lists = {k:np.array(list[v]) for k,v in adj_lists.items()}

  
    return num_nodes,feat_data,labels,len(label_id),adj_lists


PPI = "Data\\ppi"

def load_ppi():
    num_nodes = 14755
    # num_feats : 50

    feat_data = np.float32(np.load(os.path.join(PPI,"toy-ppi-feats.npy")))

    normalizer = StandardScaler()
    feat_data = normalizer.fit_transform(feat_data)

    adj_lists = defaultdict(set)

    with open(os.path.join(PPI,"toy-ppi-walks.txt")) as fe :
        for line in fe :
            edge = line.strip().split()
            item1 = int(edge[0])
            item2 = int(edge[1])
            adj_lists[item1].add(item2)
            adj_lists[item2].add(item1)
        
    adj_lists = {k:np.array(list(v)) for k,v in adj_lists.items()}

            

    

    return num_nodes,feat_data,adj_lists