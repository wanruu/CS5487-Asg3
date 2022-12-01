import tqdm
import time
import numpy as np

from utils import euclidean, centers_init


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
        print("K-means starts.")
        n = points.shape[0]

        start = time.time()
        centers = self.centers_init(points, k, self.dist_func)  # (k,d)
        labels = np.zeros(n) - 1  # (n,)
        end = time.time()
        print("Initialization done.", f"(t={end-start}s)")

        start = time.time()
        for interation in tqdm.tqdm(range(self.max_iter)):
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
        end = time.time()
        print("K-means done.", f"(t={end-start}s, iter={interation})")

        return labels



if __name__ == "__main__":
    pass

