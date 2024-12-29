import cv2

points = []


def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 7:
            points.append([x, y])

            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

            cv2.putText(img, f'({x}, {y})', (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imshow("Image", img)
        if len(points) == 7:
            print("Koordinate temena: ", points)
            osmoteme(points)


def vektorski_proizvod(p, q):
    return [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]]


def osmoteme(temena):
    t1 = temena[0] + [1]
    t2 = temena[1] + [1]
    t3 = temena[2] + [1]
    t5 = temena[3] + [1]
    t6 = temena[4] + [1]
    t7 = temena[5] + [1]
    t8 = temena[6] + [1]

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

    cv2.circle(img, rez, 5, (255, 0, 0), -1)
    cv2.putText(img, f'{rez[0]}, {rez[1]}', (rez[0]+10, rez[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow("Image", img)
    print("Osma koordinata: ", rez)
    return rez


img = cv2.imread('slika.jpg')
cv2.imshow("Image", img)

cv2.setMouseCallback("Image", select_point)  # nama ce mis postati cetkica za crtanje sad

cv2.waitKey(0)
cv2.destroyAllWindows()





