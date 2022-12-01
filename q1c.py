import numpy as np
import matplotlib.pyplot as plt

from utils import DATA_Q1
from mean_shift import MeanShift



for dataset in DATA_Q1:
    points = DATA_Q1[dataset]["X"].T  # (2,200)

    cluster_nums = []
    hs = np.arange(0.1, 10, 0.1)
    for h in hs:
        ms = MeanShift(h)
        labels = ms.fit(points)
        cluster_nums.append(len(set(labels)))

    fig = plt.figure(dpi=300)
    plt.xlabel("bandwidth h")
    plt.ylabel("cluster number")
    plt.plot(hs, cluster_nums)
    fig.savefig(f"p1-imgs/{dataset}-ms-senstive")