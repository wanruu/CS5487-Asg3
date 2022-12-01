import numpy as np
from utils import DATA_Q1, plot, euclidean, centers_init


class KMeans:
    def __init__(self, max_iter=1000, center_init="default", dist_func="euclidean"):
        self.max_iter = max_iter
        if dist_func == "euclidean":
            self.dist_func = euclidean
        if center_init == "default":
            self.centers_init = centers_init

    def fit(self, points, k):
        """
        points: (n,d)
        """
        n = points.shape[0]

        centers = self.centers_init(points, k, self.dist_func)  # (k,d)
        labels = np.zeros(n) - 1  # (n,)

        for _ in range(self.max_iter):
            last_labels = np.copy(labels)

            # cluster assignment
            for idx in range(n):
                distance = [self.dist_func(points[idx], center) for center in centers]
                labels[idx] = np.argmin(distance)

            # estimate center
            for cluster in range(k):
                c_points = np.array([points[p_idx] for p_idx in range(n) if labels[p_idx] == cluster])
                centers[cluster] = np.mean(c_points, axis=0)

            if np.all(last_labels == labels):
                break

        return labels



if __name__ == "__main__":

    for dataset in DATA_Q1:
        points = DATA_Q1[dataset]["X"].T  # (2,200)

        kmeans = KMeans()
        labels = kmeans.fit(points, 4)
        plot(points, labels, f"{dataset}-kmeans")

