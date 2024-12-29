# Napisati Python3 funkciju DLT(origs, imgs) koja za
# dve liste od po n tačaka vraća matricu projektivnog preslikavanja koje prvih
# n tačaka približno preslikava redom u drugih n.
# Pri tom se koristi SVD dekompozicija matrice za odredjivanje optimalnog resenja.

import numpy as np
from numpy import linalg

np.set_printoptions(precision=5, suppress=True)


# pomocne


def DLT(origs, imgs):
    x = origs[0][0]
    y = origs[0][1]
    z = origs[0][2]

    u = imgs[0][0]
    v = imgs[0][1]
    w = imgs[0][2]

    sistem = np.array([
        [0, 0, 0, -w * x, -w * y, -w * z, v * x, v * y, v * z],
        [w * x, w * y, w * z, 0, 0, 0, -u * x, -u * y, -u * z]
    ])

    for i in range(1, len(origs)):
        x = origs[i][0]
        y = origs[i][1]
        z = origs[i][2]

        u = imgs[i][0]
        v = imgs[i][1]
        w = imgs[i][2]

        row1 = np.array([0, 0, 0, -w * x, -w * y, -w * z, v * x, v * y, v * z])
        row2 = np.array([w * x, w * y, w * z, 0, 0, 0, -u * x, -u * y, -u * z])

        sistem = np.vstack((sistem, row1))
        sistem = np.vstack((sistem, row2))

    U, S, V = np.linalg.svd(sistem)
    mat = V[-1].reshape(3, 3)
    mat = mat/mat[2, 2]

    return mat
