# Napisati Python3 funkciju DLTwithNormalization(origs, imgs)
# koja za dve liste od po n tačaka vraća matricu projektivnog preslikavanja koje prvih n tačaka
# približno preslikava redom u drugih n. Prvo se  vrši normalizacija originala i slika,
# a zatim na normalizovane tačke  primenjuje običan DLT algoritam (prethodni zadaci).

import numpy as np
from numpy import linalg

np.set_printoptions(precision=5, suppress=True)
import math


# pomocne funkcije

def normMatrix(points):
    # kod

    # teziste formula
    x = sum([p[0] / p[2] for p in points]) / len(points)
    y = sum([p[1] / p[2] for p in points]) / len(points)

    # srednje rastojanje
    r = 0.0

    for i in range(len(points)):
        # translacija kordinatnog pocetka
        t1 = float(points[i][0] / points[i][2]) - x
        t2 = float(points[i][1] / points[i][2]) - y

        r = r + math.sqrt(t1 ** 2 + t2 ** 2)

    r = r / float(len(points))

    # skaliranje
    S = float(math.sqrt(2)) / r

    mat = np.array([[S, 0, -S * x], [0, S, -S * y], [0, 0, 1]])
    return mat


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
    mat = mat / mat[2, 2]

    return mat


def DLTwithNormalization(origs, imgs):

    # transformacije
    T = normMatrix(origs)
    T_prim = normMatrix(imgs)

    # normalizacije tacke
    M_line = T.dot(np.transpose(origs))
    M_prim = T_prim.dot(np.transpose(imgs))

    M_line = np.transpose(M_line)
    M_prim = np.transpose(M_prim)

    P_line = DLT(M_line, M_prim)
    m = (np.linalg.inv(T_prim)).dot(P_line).dot(T)

    if m[2][2] != 0 and m[2][2] != 1:
        m = m/m[2][2]

    return m

trapez = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1], [1, 2, 3], [-8, -2, 1]]
pravougaonik1 = [[- 2, - 1, 1], [2, - 1, 1], [2, 1, 1], [- 2, 1, 1], [2, 1, 5], [-16, -5, 5]]
print(DLTwithNormalization(trapez, pravougaonik1))
