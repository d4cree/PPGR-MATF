import numpy as np
from numpy import linalg
import math
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)

size = 1

cube_points = np.array([[-size, -size, -size],
                        [size, -size, -size],
                        [size, size, -size],
                        [-size, size, -size],
                        [-size, -size, size],
                        [size, -size, size],
                        [size, size, size],
                        [-size, size, size]])

edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
         [0, 4], [1, 5], [2, 6], [3, 7]]

org1 = np.array([1, -1, -1]) * (np.array([1600, 0, 0]) - np.array(
    [[608, 653, 1], [789, 841, 1], [574, 1010, 1], [382, 801, 1], [381, 938, 1], [546, 1123, 1], [740, 969, 1],
     [586, 884, 1]]))

imgs1 = np.array([[0, 0, 3, 1], [0, 3, 3, 1], [3, 3, 3, 1], [3, 0, 3, 1], [3, 0, 0, 1], [3, 3, 0, 1], [0, 3, 0, 1],
                  [2, 2, 3, 1]])

def dveJednacine(img, org):
    nula = np.array([0, 0, 0, 0])
    prva = np.array(np.concatenate((nula, -img[2] * org, img[1] * org)))
    druga = np.array(np.concatenate((img[2] * org, nula, -img[0] * org)))
    return [prva, druga]


def napraviMatricu(imgs, origs):
    A = []
    for i in range(len(imgs)):
        img = imgs[i]
        org = origs[i]
        A.extend(dveJednacine(img, org))

    return A


def matricaKamere(pts2D, pts3D):
    A = napraviMatricu(pts2D, pts3D)
    A = np.array(A)

    U, S, Vh = np.linalg.svd(A)
    T = Vh[11]
    T = T / T[11]
    T = T.reshape(3, 4)
    T = np.where(np.isclose(T, 0), 0.0, T)
    return T


def centar(T):
    C1 = np.linalg.det(np.delete(T, 0, 1))
    C2 = np.linalg.det(np.delete(T, 1, 1))
    C3 = np.linalg.det(np.delete(T, 2, 1))
    C4 = np.linalg.det(np.delete(T, 3, 1))
    C = np.array([C1, -C2, C3, -C4]) / (-C4)
    C = np.where(np.isclose(C, 0), 0.0, C)

    return C


def kameraK(T):
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


T1 = matricaKamere(org1, imgs1)
print("Matrica kamere: \n", T1)
print()
print("Matrica kalibracije kamere: \n", kameraK(T1))
print()
print("Pozicija kamere:\n", centar(T1))
print()
print("Spoljasnja matrica kamere:\n", kameraA(T1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for edge in edges:
    ax.plot3D(*zip(*cube_points[edge]), color="b")

ax.quiver(0, 0, 0, size, 0, 0, color="red", label="World X")
ax.quiver(0, 0, 0, 0, size, 0, color="green", label="World Y")
ax.quiver(0, 0, 0, 0, 0, size, color="blue", label="World Z")

camera_center = centar(T1)
small_size = 0.5
camera_square = small_size * np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
camera_square += camera_center[:3] #translacija
square_edges = [[0, 1], [1, 2], [2, 3], [3, 0]]

A1 = kameraA(T1)
camera_axes = np.transpose(A1)
for i, color, label in zip(range(3), ["orange", "purple", "cyan"], ["Camera X", "Camera Y", "Camera Z"]):
    ax.quiver(
        camera_center[0], camera_center[1], camera_center[2],
        camera_axes[i, 0], camera_axes[i, 1], camera_axes[i, 2],
        color=color, label=label
    )

for i in range(len(camera_square)):
    p1 = camera_square[i]
    p2 = camera_square[(i + 1) % len(camera_square)]
    ax.plot3D(*zip(p1, p2), color="black", linestyle="dashed", linewidth=1.5)

ax.set_xlim([-5, 7])
ax.set_ylim([-5, 7])
ax.set_zlim([-5, 7])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
