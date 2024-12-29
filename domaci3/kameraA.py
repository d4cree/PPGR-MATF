import numpy as np
from numpy import linalg
import math

np.set_printoptions(precision=5, suppress=True)


# ovde pišete pomocne funkcije

def kameraA(T):
    t0 = np.delete(T, 3, 1)

    if np.linalg.det(t0) < 0:
        T = -T
        t0 = np.delete(T, 3, 1)

    t0i = np.linalg.inv(t0)
    Q, R = np.linalg.qr(t0i)

    if R[0, 0] < 0:
        R = np.matmul(np.diag([-1, 1, 1]), R)
        Q = np.matmul(Q, np.diag([-1, 1, 1]))

    if R[1, 1] < 0:
        R = np.matmul(np.diag([1, -1, 1]), R)
        Q = np.matmul(Q, np.diag([1, -1, 1]))

    if R[2, 2] < 0:
        R = np.matmul(np.diag([1, 1, -1]), R)
        Q = np.matmul(Q, np.diag([1, 1, -1]))

    A = Q
    A = np.where(np.isclose(A, 0), 0.0, A)

    return A


T = np.array([[-2, 3, 0, 7], [-3, 0, 3, -6], [1, 0, 0, -2]])

print(kameraA(T))
