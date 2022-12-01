import numpy as np
from utils import DATA_Q1, plot
import matplotlib.pyplot as plt

SHIFT_THRESHOLD = 1e-6
CLUSTER_THRESHOLD = 1e-1


class MeanShift:
    def __init__(self, bandwidth, kernel="gaussian"):
        self.bandwidth = bandwidth
        if kernel == "gaussian":
            self.kernel = self.gaussian_kernel


    def gaussian_kernel(self, center: np.array, points: np.array):
        """
        center: (d,)
        points: (n,d)
        """
        n, _ = points.shape
        # distances = np.array([np.linalg.norm(center - points[i]) for i in range(n)])
        distances = np.sqrt(((center - points)**2).sum(axis=1))
        weights = (1/(self.bandwidth*np.sqrt(2*np.pi))) * np.exp(-0.5*distances**2/self.bandwidth**2)
        return weights  # (n,)


    def _shift_point(self, point: np.array, points: np.array):
        n, _ = points.shape
        weights = self.kernel(point, points)  # (n,)
        weights = weights.reshape((n, -1))  # (n,1)
        offset = points * weights  # (n,d)
        offset = np.sum(offset, axis=0)  # (d,)
        scale = np.sum(weights)
        return offset / scale  # (d,)


    def fit(self, points: np.array):
        n, d = points.shape

        shift_points = np.array(points)
        if_shift = [True for _ in range(n)]

        # shift
        max_shift_dist = None
        while max_shift_dist is None or max_shift_dist > SHIFT_THRESHOLD:
            max_shift_dist = 0
            for idx in range(n):
                if not if_shift[idx]:
                    continue
                # calculate shift point
                origin_point = shift_points[idx]
                shift_point = self._shift_point(origin_point, points)
                # shift distance
                shift_dist = np.linalg.norm(origin_point - shift_point)
                max_shift_dist = max(max_shift_dist, shift_dist)
                if_shift[idx] = shift_dist > SHIFT_THRESHOLD
                # shift
                shift_points[idx] = shift_point

        # cluster
        cluster_ids = [0]
        cluster_centers = [shift_points[0]]
        cluster_idx = 1

        for idx in range(1, n):
            point = shift_points[idx]

            for center_idx, center in enumerate(cluster_centers):
                dist = np.linalg.norm(center - point)
                if dist < CLUSTER_THRESHOLD:
                    cluster_ids.append(center_idx)
                    break
            if len(cluster_ids) <= idx:
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
        return np.array(cluster_ids)



if __name__ == "__main__":
    # h = 1.7

    for dataset in DATA_Q1:
        points = DATA_Q1[dataset]["X"].T  # (2,200)

        cluster_nums = []
        hs = np.arange(1.1, 10, 0.1)
        for h in hs:
            ms = MeanShift(h)
            labels = ms.fit(points)
            cluster_num = len(set(labels))
            cluster_nums.append(cluster_num)
            # print(h, len(set(labels)))

        # plot(points, labels, f"{dataset}-ms")
        fig = plt.figure(dpi=300)
        plt.xlabel("bandwidth h")
        plt.ylabel("cluster number")
        plt.plot(hs, cluster_nums)
        fig.savefig(f"p1-imgs/{dataset}-ms-senstive")


        # Testing meanshift.
        # from sklearn.cluster import MeanShift
        # clustering = MeanShift(bandwidth=3.7).fit(X.T)
        # plot(X, clustering.labels_, f"{dataset}-test-ms")

