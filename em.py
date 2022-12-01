import numpy as np
from utils import DATA_Q1, plot, euclidean, centers_init


class EM:
    def __init__(self, center_init="default", dist_func="euclidean", model="mv_gaussian"):
        if dist_func == "euclidean":
            self.dist_func = euclidean
        if center_init == "default":
            self.centers_init = centers_init
        if model == "mv_gaussian":
            self.prob = self.mv_gaussian
            self.ll = self.mv_gaussian_ll



    def mv_gaussian(self, point, mean, covar, log=True):
        d = point.size
        coeff = 1 / np.power(2 * np.pi, d / 2) / np.power(np.linalg.det(covar), 0.5)
        exp = -0.5 * (point - mean).dot(np.linalg.inv(covar)).dot(point - mean)
        if log:
            return np.log(coeff) + exp
        return coeff * np.exp(exp)

    def mv_gaussian_ll(self, points, pis, means, covars):
        result = 0
        for point in points:
            result += sum([np.log(pis[j]) + self.mv_gaussian(point, means[j], covars[j], True) for j in range(pis.size)])
        return result


    def fit(self, points, k):
        # initialize
        n, d = points.shape
        pis = np.ones(k) / k  # (k,)
        means = centers_init(points, k, self.dist_func)  # (k,d)
        covars = [np.identity(d) for _ in range(k)]  # (k,d,d)

        # log-likelihood
        last_ll = self.ll(points, pis, means, covars)

        interations = 0
        while True:
            # e-step: calculate z
            l_ij = [[np.log(pis[j]) + self.prob(points[i], means[j], covars[j], True) for j in range(k)] for i in range(n)]  # (n,k)
            l_ij = np.array(l_ij)
            
            z = np.zeros((n,k))
            for i in range(n):
                # l_ij[i]: (k,)
                max_l = l_ij[i].max()
                l = max_l + np.log(np.exp(l_ij[i] - max_l).sum())
                z[i] = np.exp(l_ij[i] - l)

            # m-step
            for j in range(k):
                N_j = z[:,j].sum()  # c
                pis[j] = N_j / n  # c
                means[j] = points.T.dot(z[:,j]) / N_j  # (d,)
                tmp_sum = np.zeros((d, d))
                for i in range(n):
                    sub_item = (points[i] - means[j]).reshape((d,1))
                    tmp_sum += z[i][j] * sub_item.dot(sub_item.T)
                covars[j] = tmp_sum / N_j

            # log-likelihood
            cur_ll = self.ll(points, pis, means, covars)
            print(interations, abs(last_ll - cur_ll))
            if abs(last_ll - cur_ll) < 1e-5:
                break
            last_ll = cur_ll

            interations += 1


        labels = np.array([np.argmax(z[i,:]) for i in range(n)])

        return labels


if __name__ == "__main__":
    for dataset in DATA_Q1:
        points = DATA_Q1[dataset]["X"].T  # (2,200)

        em = EM()
        labels = em.fit(points, 4)
        plot(points, labels, f"{dataset}-em")

        # Testing
        # from sklearn.mixture import GaussianMixture

        # gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0)
        # gmm.fit(points)
        # labels = gmm.predict(points)
        # plot(points.T, labels, f"{dataset}-test-em")




