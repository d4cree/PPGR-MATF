import numpy as np
from numpy import linalg
import math

np.set_printoptions(precision=5, suppress=True)


# ovde pišete pomocne funkcije

def dveJednacine(img, org):
    nula = np.array([0, 0, 0, 0])
    prva = np.array(np.concatenate((nula, -img[2]*org, img[1]*org)))
    druga = np.array(np.concatenate((img[2]*org, nula, -img[0] * org)))
    return [prva, druga]

def napraviMatricu(imgs, origs):
    A = []
    for i in range(len(imgs)):
        img = imgs[i]
        org = origs[i]
        A.extend(dveJednacine(img, org))

    return A

def matricaKamere(pts2D, pts3D):
    # vaš kod
    A = napraviMatricu(pts2D, pts3D)
    A = np.array(A)

    U, S, Vh = np.linalg.svd(A)
    T = Vh[11]
    T = T/T[11]
    T = T.reshape(3, 4)
    T = np.where(np.isclose(T, 0), 0.0, T)
    return T

pts2D = np.array([[12, 61, 31], [1, 95, 4], [20, 82, 19], [56, 50, 55], [32, 65, 84], [46, 39, 16], [67, 63, 78]])
pts3D = np.array([[44, 61, 31, 99], [17, 84, 40, 45], [20, 59, 65, 3], [37, 81, 70, 82], [7, 95, 8, 29], [31, 61, 91, 37], [82, 99, 80, 7]])
print(matricaKamere(pts2D,pts3D))