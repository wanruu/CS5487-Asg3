import helper
from k_means import KMeans
from em import EM
from mean_shift import MeanShift
from utils import DATA_Q2, ID_Q2

import os
import pylab as pl


IMG_PATH = "p2-imgs"
if not os.path.exists(IMG_PATH):
    os.mkdir(IMG_PATH)


def plot():
    # draw original image
    pl.subplot(1,3,1)
    pl.imshow(img)

    # make segmentation image from labels
    segm = helper.labels2seg(labels, L)
    pl.subplot(1,3,2)
    pl.imshow(segm)

    # color the segmentation image
    csegm = helper.colorsegms(segm, img)
    pl.subplot(1,3,3)
    pl.imshow(csegm)


DATA_Q2 = DATA_Q2[:1]
for idx, img in enumerate(DATA_Q2):
    # create directory
    save_path = f"{IMG_PATH}/{ID_Q2[idx]}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # extract feature
    X, L = helper.getfeatures(img, 7)  # X: (4, 3060)

    # kmeans, em
    for k in range(20, 30):
        # kmeans
        kmeans = KMeans()
        labels = kmeans.fit(X.T, k) + 1
        plot()
        pl.savefig(f"{save_path}/kmeans(k={k})")

        # em
        # em = EM()
        # labels = em.fit(X.T, k) + 1
        # plot()
        # pl.savefig(f"{save_path}/em(k={k})")


    # # meanshift
    # for h in range(1, 10):
    #     # meanshift
    #     mean_shift = MeanShift(bandwidth=h)
    #     labels = mean_shift.fit(X.T) + 1
    #     name = "meanshift"
    #     plot()
    #     pl.savefig(f"{save_path}/meanshift(h={h})")


