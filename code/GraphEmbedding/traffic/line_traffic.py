
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def get_embedding_data(embeddings):
    emb_list = []
    for k in range(249):
        emb_list.append(embeddings[str(k + 1)])
    emb_list = np.array(emb_list)
    return emb_list

if __name__ == "__main__":
    G = nx.read_edgelist('../data/road_dist/sensor.txt', delimiter=',',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', float)])

    model = LINE(G, embedding_size=128, order='second')
    model.train(batch_size=1024, epochs=2, verbose=2)
    embeddings = model.get_embeddings()

    emb_list = get_embedding_data(embeddings)
    # write to pickle
    with open('../data/road_dist/sensor_mx.pkl', 'wb') as f:
        pickle.dump(emb_list, f)
    print('finished')
