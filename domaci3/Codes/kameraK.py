import numpy as np
from numpy import linalg
import math

np.set_printoptions(precision=5, suppress=True)


# ovde pišete pomocne funkcije

def kameraK(T):
    # vaš kod
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

    K = np.linalg.inv(R)
    K = K / K[2, 2]
    K = np.where(np.isclose(K, 0), 0.0, K)

    return K

T = np.array([[-2,3,0,7], [-3,0,3,-6], [1,0,0,-2]])
print(kameraK(T))