# Napisati Python3 funkciju naivni(origs, imgs) koja za dve liste od po 4 tačke vraća matricu
# projektivnog preslikavanja koje prve četiri tačke preslikava redom u druge četiri.

import numpy as np

np.set_printoptions(precision=5, suppress=True)


# pomocne funkcije
def find_matrix(points):
    matrix = np.array([
        [points[0][0], points[1][0], points[2][0]],
        [points[0][1], points[1][1], points[2][1]],
        [points[0][2], points[1][2], points[2][2]]
    ])

    D = np.array([points[3][0], points[3][1], points[3][2]])
    res = np.linalg.solve(matrix, D)

    alpha = res[0]
    beta = res[1]
    gamma = res[2]

    col1 = np.array([alpha * points[0][0], alpha * points[0][1], alpha * points[0][2]])
    col2 = np.array([beta * points[1][0], beta * points[1][1], beta * points[1][2]])
    col3 = np.array([gamma * points[2][0], gamma * points[2][1], gamma * points[2][2]])

    P = np.column_stack([col1, col2, col3])

    return P


def general_position(matrix):
    A = matrix[0]
    B = matrix[1]
    C = matrix[2]
    D = matrix[3]

    m1 = [A, B, C]
    m2 = [A, B, D]
    m3 = [A, C, D]
    m4 = [B, C, D]

    # determinante
    det1 = np.linalg.det(m1)
    det2 = np.linalg.det(m2)
    det3 = np.linalg.det(m3)
    det4 = np.linalg.det(m4)

    if 0 < det1 < 0.000001:
        det1 = 0
    if 0 < det2 < 0.000001:
        det2 = 0
    if 0 < det3 < 0.000001:
        det3 = 0
    if 0 < det4 < 0.000001:
        det4 = 0
    if det1 == 0 or det2 == 0 or det3 == 0 or det4 == 0:
        return False
    return True


def naivni(origs, imgs):
    if not general_position(origs):
        return "Losi originali!"
    if not general_position(imgs):
        return "Lose slike!"

    P1 = find_matrix(origs)
    P2 = find_matrix(imgs)

    P = np.dot(P2, np.linalg.inv(P1))

    # negativna nula
    for i in range(3):
        for j in range(3):
            if (P[i][j] < 0 and P[i][j] > -0.000001):
                P[i][j] = 0

    if (P[2][2] != 1 and P[2][2] != 0):
        P = P / P[2][2]

    return P
