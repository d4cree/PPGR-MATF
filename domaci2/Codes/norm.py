# Napisati Python3 funkciju normMatrix(points) koja za listu od  n tačaka ravni
# (datim homogenim koordinatama) vraća 3x3 matricu normalizacije tih tačaka.
# To je 3x3 matrica (kompozicija translacije i homotetije)
# tako da kada tačke preslikamo tom matricom, dobijamo tačke čije
# je težište u koordinatnom početku sa prosečnim rastojanjem sqrt(2) od koordinatnog početka.
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

trapez = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1], [1,2,3], [-8,-2,1]]
print(normMatrix(trapez))