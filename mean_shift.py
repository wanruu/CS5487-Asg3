import time
import numpy as np

SHIFT_THRESHOLD = 1e-6
CLUSTER_THRESHOLD = 1e-1


class MeanShift:
    def __init__(self, bandwidth, kernel="gaussian"):
        self.bandwidth = bandwidth
        if kernel == "gaussian":
            self.kernel = self.gaussian_kernel
        elif kernel == "q2b":
            self.kernel = self.meanshift_kernel_q2


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

    def meanshift_kernel_q2(self, center: np.array, points: np.array):
        """
        center: (4,)
        points: (3060, 4)
        """
        n, _ = points.shape
        hc, hp = self.bandwidth

        centers = np.split(center, 2)  # (2,2)
        pointss = [item.T for item in np.split(points.T, 2)]  # (2,3060,2)
        
        dist1 = np.sqrt(((centers[0] - pointss[0])**2).sum(axis=1))  # (3060,)
        dist2 = np.sqrt(((centers[1] - pointss[1])**2).sum(axis=1))  # (3060,)
        exp = -dist1**2/2/hc**2 - dist2**2/2/hp**2
        weights = (1 / np.pi**2 / hc**2 / hp**2) * np.exp(exp)
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
        print("Mean-shift starts.")
        start = time.time()
        
        n, d = points.shape

        shift_points = np.array(points)
        if_shift = [True for _ in range(n)]

        # shift
        interation = 0
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
            interation += 1
            if interation % 100 == 0:
                print(f"iter={interation}, max_shift_dist={max_shift_dist}")

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


        end = time.time()
        print("Mean-shift done.", f"(t={end-start}s, iter={interation}, cluster={cluster_idx})")

        return np.array(cluster_ids)



if __name__ == "__main__":
    pass
    # Testing meanshift.
    # from sklearn.cluster import MeanShift
    # clustering = MeanShift(bandwidth=3.7).fit(X.T)

