import helper
from k_means import KMeans
from em import EM
from mean_shift import MeanShift
from utils import DATA_Q2_RAW


import pylab as pl

for img in DATA_Q2_RAW:
    # original
    pl.subplot(1,3,1)
    pl.imshow(img)
    X, L = helper.getfeatures(img, 7)  # X: (4, 3060)
    K = 7

    # kmeans
    # kmeans = KMeans()
    # labels = kmeans.fit(X.T, K) + 1


    # em
    em = EM()
    labels = em.fit(X.T, K) + 1

    # # meanshift
    # mean_shift = MeanShift(bandwidth=2)
    # labels = mean_shift.fit(X.T) + 1

    # make segmentation image from labels
    segm = helper.labels2seg(labels, L)
    pl.subplot(1,3,2)
    pl.imshow(segm)

    # color the segmentation image
    csegm = helper.colorsegms(segm, img)
    pl.subplot(1,3,3)
    pl.imshow(csegm)


    pl.show()
    break