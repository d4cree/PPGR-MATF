def vektorski_proizvod(p, q):
    return [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]]


def osmoteme(temena):
    t7 = temena[0] + [1]
    t6 = temena[1] + [1]
    t5 = temena[2] + [1]
    t8 = temena[3] + [1]
    t3 = temena[4] + [1]
    t2 = temena[5] + [1]
    t1 = temena[6] + [1]

    t23 = vektorski_proizvod(t2, t3)
    t67 = vektorski_proizvod(t6, t7)
    x = vektorski_proizvod(t23, t67)

    ################################

    t65 = vektorski_proizvod(t6, t5)
    t78 = vektorski_proizvod(t7, t8)
    y = vektorski_proizvod(t65, t78)

    ################################

    xt1 = vektorski_proizvod(x, t1)
    t3y = vektorski_proizvod(t3, y)
    t4 = vektorski_proizvod(xt1, t3y)
    rez = [int(t4[0] / t4[2]), int(t4[1] / t4[2])]

    return rez

rez = osmoteme([[32, 70], [195, 144], [195, 538], [30, 307], [251, 40], [454, 78], [455, 337]])
print(rez)