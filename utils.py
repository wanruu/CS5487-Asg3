import re
import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex


import helper


_data = loadmat("PA2-cluster-data/cluster_data.mat")
DATA_Q1 = {
    dataset: {
        datatype: _data[f"data{dataset}_{datatype}"] for datatype in ["X", "Y"]
    } for dataset in ["A", "B", "C"]
}


_filenames = ["images/"+f for f in os.listdir("images/") if f[-4:] == ".jpg"]
DATA_Q2 = [Image.open(f) for f in _filenames]
ID_Q2 = [re.search(r"(\d+)\.jpg", f).group(1) for f in _filenames]


def plot(points, labels, title, centers=[], dirname="p1-imgs"):
    fig = plt.figure(dpi=300)
    plt.title(title)

    n = points.shape[0]
    colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for _ in range(n)])
    colors = [rgb2hex(x) for x in colors]

    for idx, cluster in enumerate(list(labels)):
        plt.scatter([points[idx][0]], [points[idx][1]], c=colors[int(cluster)])

    fig.savefig(f"{dirname}/"+title)



def euclidean(x1, x2):
    return np.linalg.norm(x1 - x2)


def centers_init(points: np.array, k: int, dist_func) -> np.array:
    """
    points: (n,d)
    return: (k,d)
    """
    n = points.shape[0]
    if k > n:
        raise Exception("Cluster number larger than data size")

    # Find index of cluster center
    selected_idxes = np.random.choice(n, 1).tolist()  # randomly choose a point

    while len(selected_idxes) < k:
        # cal dist between each point and its nearest cluster
        distances = []
        for p_idx in range(n):
            distance = [dist_func(points[p_idx], points[c_idx]) for c_idx in selected_idxes]
            distances.append(min(distance))
        selected_idxes.append(np.argmax(np.array(distances)))

    selected_idxes = np.array(selected_idxes)

    return points[selected_idxes]

