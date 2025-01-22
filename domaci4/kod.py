import numpy as np
from numpy import linalg
import math
import plotly.graph_objects as go
import plotly.express as px


def piksel(tacka):
    x, y, z = tacka
    return [1600 - x, y, z]

# piksel koordinate
P1L = piksel([668, 675, 1])
P2L = piksel([825, 642, 1])
P3L = piksel([729, 558, 1])
P4L = piksel([569, 583, 1])
P5L = piksel([638, 861, 1])
P6L = piksel([798, 810, 1])
P7L = piksel([704, 710, 1])
P8L = piksel([546, 754, 1])

Q1L = piksel([876, 639, 1])
Q2L = piksel([973, 588, 1])
Q3L = piksel([800, 540, 1])
Q4L = piksel([821, 460, 1])
Q5L = piksel([988, 492, 1])
Q6L = piksel([925, 420, 1])

R1L = piksel([1225, 682, 1])
R2L = piksel([1340, 641, 1])
R3L = piksel([1149, 591, 1])
R4L = piksel([1291, 492, 1])

P1D = piksel([540, 540, 1])
P2D = piksel([740, 533, 1])
P3D = piksel([689, 448, 1])
P4D = piksel([494, 458, 1])
P5D = piksel([519, 749, 1])
P6D = piksel([714, 709, 1])
P7D = piksel([661, 604, 1])
P8D = piksel([489, 623, 1])

Q1D = piksel([893, 572, 1])
Q2D = piksel([1039, 533, 1])
Q3D = piksel([866, 487, 1])
Q4D = piksel([860, 383, 1])
Q5D = piksel([1051, 424, 1])
Q6D = piksel([1000, 359, 1])


R1D = piksel([1289, 668, 1])
R2D = piksel([1472, 649, 1])
R3D = piksel([1245, 567, 1])
R4D = piksel([1407, 466, 1])

leve8 = np.array([P4L, P3L, P2L, P1L, R1L, R2L, R3L, R4L])
desne8 = np.array([P4D, P3D, P2D, P1D, R1D, R2D, R3D, R4D])


leve8 = np.array(leve8, dtype=float)
desne8 = np.array(desne8, dtype=float)

leve = np.array([
    P4L, P3L, P2L, P1L, P7L, P6L, P5L, P8L,
    Q1L, Q2L, Q3L, Q4L, Q5L, Q6L,
    R1L, R2L, R3L, R4L
])
desne = np.array([
    P4D, P3D, P2D, P1D, P7D, P6D, P5D, P8D,
    Q1D, Q2D, Q3D, Q4D, Q5D, Q6D,
    R1D, R2D, R3D, R4D
])

# Fundamentalna matrica F
def fundamentalna(u, v):
    a1, a2, a3 = u
    b1, b2, b3 = v

    F = [ 
        a1*b1, a2*b1, a3*b1,
        a1*b2, a2*b2, a3*b2,
        a1*b3, a2*b3, a3*b3
    ]
    return F

jed8 = np.array([fundamentalna(u, v) for u, v in zip(leve8, desne8)])
print(jed8)

U, S, V = np.linalg.svd(jed8)
n = len(V)


Fvector = V[n-1]
FF = np.array([Fvector[i:i+3] for i in range(0, len(Fvector), 3)])
print("FF",FF)

# odredjivanje epipolova
U, DD, V = np.linalg.svd(FF)
e1 = V[2, :]
e1 = (1/e1[2])*e1

e2 = U[:, 2]
e2 = (1/e2[2])*e2

diag1 = np.diag([1, 1, 0])
DD1 = np.dot(diag1, np.diag(DD))


# ispravka FF matrice
FF1 = np.dot(U, np.dot(DD1, V))
print("FF1: ", FF1)

# osnovna matrica E
K1 = np.array([
    [1300, 0, 800],
    [0, 1300, 600],
    [0, 0, 1]
])
K1T = np.transpose(K1)

EE = np.dot(np.dot(K1T, FF1), K1)
print("EE:\n", EE)

# dekompozicija EE
Q0 = [
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
]
E0 = [
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 0]
]
print("E0: ", E0)

U, SS, V = np.linalg.svd(EE)
V = -(np.transpose(V))
EC = np.dot(np.dot(U, E0), np.transpose(U))
AA = np.dot(np.dot(U, Q0), np.transpose(V))

def koso2v(A):
    return np.array([A[2, 1], A[0, 2], A[1, 0]])

CC = koso2v(EC)
print("EC", EC)
print("AA", AA)

# matrice kamera u koordinatnom sistemu druge kamere
T2 = np.array([
    [1300, 0, 800, 0],
    [0, 1300, 600, 0],
    [0, 0, 1, 0]
])


CC1 = (np.dot(-(np.transpose(AA)), CC))
print("CC1:")
print(CC1)

temp1 = np.transpose(np.dot(K1, np.transpose(AA)))
T1 = np.transpose(np.vstack([temp1, np.dot(K1, CC1)]))

# Triangulacija (opsti slucaj)

T1p = np.array([[-2, -1, 0, 2], [-3, 0, 1, 0], [-1, 0, 0, 0]])
T2p = np.array([[2, -2, 0, -2], [0, -3, 2, -2], [0, -1, 0, 0]])
M1 = np.array([5, 3, 1])
M2 = np.array([-2, 1, 1])

def jednacine(T1, T2, m1, m2):

    eq1 = m1[1] * T1[2] - m1[2] * T1[1]
    eq2 = -m1[0] * T1[2] + m1[2] * T1[0]
    eq3 = m2[1] * T2[2] - m2[2] * T2[1]
    eq4 = -m2[0] * T2[2] + m2[2] * T2[0]
    
    return np.array([eq1, eq2, eq3, eq4])

def UAfine(XX):
    XX = np.array(XX)
    normalizovano = XX / XX[-1]
    return normalizovano[:-1]


def triang(T1, T2, M1, M2):
    jedM = jednacine(T1, T2, M1, M2)
    _, _, V = np.linalg.svd(jedM)
    rez = UAfine(V[3])
    return rez

# Triangulacija tacaka sa fotografije
print("T1:")
print(T1)
print("T2:")
print(T2)


tacke3D = np.array(list(map(lambda x1, x2: triang(T1, T2, x1, x2), leve, desne)))

print("3D Tacke:")
for i, point in enumerate(tacke3D):
    print(f"Point {i+1}: {point}")

temenaKocke = np.array(tacke3D)
ivice = [[0, 1], [0, 3], [0, 7],
         [1, 2], [1, 4],
         [2, 3], [2, 5],
         [3, 6],
         [4, 5], [4, 7],
         [5, 6],
         [6, 7],
         [8, 9], [8, 10], [8,11], [8, 12],
         [9, 12], [9, 10], [9, 13],
         [10, 11], [10, 13],
         [11, 12], [11, 13],
         [12, 13],
         [14, 15], [14, 16], [14, 17],
         [15, 16], [15, 17],
         [16, 17]
        ]


def prikazKocke(): 
    # izdvajamo x,y,z koordinate svih tacaka
    xdata = (np.transpose(temenaKocke))[0]
    ydata = (np.transpose(temenaKocke))[1]
    zdata = (np.transpose(temenaKocke))[2]
    # u data1 ubacujemo sve sto treba naccrtati
    data1 = []
    # za svaku ivicu crtamo duz na osnovu koordinata
    for i in range(len(ivice)):
        data1.append(go.Scatter3d(x=[xdata[ivice[i][0]], xdata[ivice[i][1]]], y=[ydata[ivice[i][0]], ydata[ivice[i][1]]],z=[zdata[ivice[i][0]], zdata[ivice[i][1]]]))
    fig = go.Figure(data = data1 )
    # da ne prikazuje legendu
    fig.update_layout(showlegend=False)
    fig.show()
    # pravi html fajl (ako zelite da napravite "rotatable" 3D rekonstruciju)
    # birate kao parametar velicinu apleta. fulhtml=False je vazno da ne bi pravio ogroman fajl
    # ovde stavite neki vas folder
    fig.write_html("C:/Users/heheh/PycharmProjects/PPGR_domaci2/domaci4/test.html", include_plotlyjs = 'cdn', default_width = '800px', default_height = '600px', full_html = False) #Modifiy the html file
    fig.show()

prikazKocke()
