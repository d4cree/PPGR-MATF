from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float
    z: float = 1.0

    def vektorski_proizvod(self, p, q):
        return Point(p.y * q.z - p.z*q.y, p.z * q.x - p.x * q.z, p.x * q.y - p.y * q.x)

    def nevidljiva_tacka(self, a, b, c, e, f, g, h):
        cb = self.vektorski_proizvod(c, b)

        gf = self.vektorski_proizvod(g, f)

        m = self.vektorski_proizvod(cb, gf)

        ##############################################

        fe = self.vektorski_proizvod(f, e)

        gh = self.vektorski_proizvod(g, h)

        n = self.vektorski_proizvod(fe, gh)

        ##############################################

        ma = self.vektorski_proizvod(m, a)

        cn = self.vektorski_proizvod(c, n)

        d = self.vektorski_proizvod(ma, cn)

        print("({}, {})".format(d.x / d.z, d.y / d.z))


def main():
    a = Point(751, 262)
    b = Point(542, 526)
    c = Point(279, 431)
    e = Point(759, 151)
    f = Point(541, 352)
    g = Point(249, 274)
    h = Point(546, 110)

    p = Point(0, 0)

    p.nevidljiva_tacka(a, b, c, e, f, g, h)


if __name__ == "__main__":
    main()
