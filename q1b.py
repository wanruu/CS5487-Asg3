from k_means import KMeans
from em import EM
from mean_shift import MeanShift
from utils import DATA_Q1, plot


K = 4
for dataset in DATA_Q1:
    points = DATA_Q1[dataset]["X"].T  # (2,200)

    # kmeans
    kmeans = KMeans()
    labels = kmeans.fit(points, K)
    plot(points, labels, f"{dataset}-kmeans", dirname="p1-imgs")

    # em
    em = EM()
    labels = em.fit(points, 4)
    plot(points, labels, f"{dataset}-em", dirname="p1-imgs")

    # meanshift
    cluster_nums = []
    ms = MeanShift(1.7)
    labels = ms.fit(points)
    plot(points, labels, f"{dataset}-meanshift", dirname="p1-imgs")
