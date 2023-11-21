import collections
from functools import reduce
import numpy as np
import pandas as pd
import os

def get_neigh(nodes,adj_lists):
    return np.concatenate([adj_lists[i] for i in nodes])



def diff_matrix(dst_nodes,adj_lists,sample_size):
    max_node = max(list(adj_lists.keys()))
    def sample(ns):
        return np.random.choice(ns,min(sample_size,len(ns)),replace=False)
    
    def vectorize(ns):
        v = np.zeros(max_node+1,dtype=np.float32)
        v[ns] = 1
        return v
    

    adj_mat_all= np.stack([vectorize(sample(adj_lists[i])) for i in dst_nodes])
    one_cols = np.any(adj_mat_all.astype(np.bool),axis=0)

    # compute diff matrix
    adj_mat = adj_mat_all[:,one_cols]
    adj_mat_sum = np.sum(adj_mat ,axis=1,keepdims=True)
    diff_mat = adj_mat / adj_mat_sum

    src_nodes = np.arange(one_cols.size)[one_cols]
    
    dst_src = np.union1d(dst_nodes,src_nodes)
    dst_src_src = np.searchsorted(dst_src,src_nodes)
    dst_src_dst = np.searchsorted(dst_src,dst_nodes)

    

    return dst_src,dst_src_src,dst_src_dst,diff_mat





def Build_batch_from_nodes(nodes,adj_lists,sample_sizes):
    

    
    
    dst_nodes = [nodes]
    dstsrc_srcs = []
    dstsrc_dsts = []
    diff_mats = []

    for sample_size in reversed(sample_sizes) :
        dn,dss,dsd,dm = diff_matrix(dst_nodes[-1],adj_lists,sample_size)
        dst_nodes.append(dn)
        dstsrc_dsts.append(dsd)
        dstsrc_srcs.append(dss)
        diff_mats.append(dm)
    
    src_nodes = dst_nodes.pop()

    columns = ["src_nodes","dst_src_src","dst_src_dst","dif_mats"]
    Batch = collections.namedtuple("Batch",columns)

    

    return Batch(src_nodes,dstsrc_srcs,dstsrc_dsts,diff_mats) 


def Build_batch_from_edges(edges,nodes,adj_lists,sample_sizes,neg_size):
    batchA,batchB = edges.transpose()

    possible_negs =  reduce(np.setdiff1d,(nodes,batchA,get_neigh(batchA,adj_lists),batchB,get_neigh(batchB,adj_lists)))
    
    batchN = np.random.choice(possible_negs,size=min(neg_size,len(possible_negs)),replace=False)

    batch_all = np.unique(np.concatenate([batchA,batchB,batchN]))

    dst2batchA = np.searchsorted(batch_all,batchA) #indices of BatchA in batch_all
    dst2batchB = np.searchsorted(batch_all,batchB) #indices of Batchلا in batch_all
    dst2batchN = np.in1d(batch_all,batchN) 


    res = Build_batch_from_nodes(batch_all,adj_lists,sample_sizes)

    columns = ["src_nodes","dst_src_src","dst_src_dst","dif_mats",
             "dst2batchA","dst2batchB","dst2batchN"]
    Batch = collections.namedtuple("Batch",columns)
    
    return Batch(res.src_nodes,res.dst_src_src,res.dst_src_dst,res.dif_mats,dst2batchA,dst2batchB,dst2batchN)
