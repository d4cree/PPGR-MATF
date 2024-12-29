# Napisati Python 3 funkciju centar(T)  koja za datu 3x4 matricu kamere T,
# tipa np.array, vraća homogene koordinate centra kamere (sa 1 na poslednjoj koordinati),
# 4-vektor, tipa np.array.

import numpy as np
from numpy import linalg
import math

np.set_printoptions(precision=5, suppress=True)


# ovde pišete pomocne funkcije

def centar(T):
    # vaš kod
    C1 = np.linalg.det(np.delete(T, 0, 1))
    C2 = np.linalg.det(np.delete(T, 1, 1))
    C3 = np.linalg.det(np.delete(T, 2, 1))
    C4 = np.linalg.det(np.delete(T, 3, 1))
    C = np.array([C1, -C2, C3, -C4])/(-C4)
    C = np.where(np.isclose(C, 0), 0.0, C)

    return C

T = np.array([[-2,3,0,7], [-3,0,3,-6], [1,0,0,-2]])
print(centar(T))

