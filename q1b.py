from k_means import KMeans
from em import EM
from mean_shift import MeanShift
from utils import DATA_Q1

import os
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex


IMG_PATH = "p1-imgs"
if not os.path.exists(IMG_PATH):
    os.mkdir(IMG_PATH)


def plot(points, labels, title, centers=[]):
    fig = plt.figure(dpi=300)
    plt.title(title)

    n = points.shape[0]
    colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for _ in range(n)])
    colors = [rgb2hex(x) for x in colors]

    for idx, cluster in enumerate(list(labels)):
        plt.scatter([points[idx][0]], [points[idx][1]], c=colors[int(cluster)])

    fig.savefig(f"{IMG_PATH}/"+title)



k = 4
h = 1.7
for dataset in DATA_Q1:
    points = DATA_Q1[dataset]["X"].T  # (2,200)

    # kmeans
    kmeans = KMeans()
    labels = kmeans.fit(points, k)
    plot(points, labels, f"{dataset}-kmeans")

    # em
    em = EM()
    labels = em.fit(points, k)
    plot(points, labels, f"{dataset}-em")

    # meanshift
    cluster_nums = []
    ms = MeanShift(h)
    labels = ms.fit(points)
    plot(points, labels, f"{dataset}-meanshift", dirname="p1-imgs")


