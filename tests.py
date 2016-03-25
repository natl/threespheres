from __future__ import division, print_function, unicode_literals

import unittest
import threespheres

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA


def triple_sphere_plot(p1, p2, p3, r1, r2, r3):
    """
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (r, p, c) in zip([r1, r2, r3], [p1, p2, p3], ['r', 'g', 'b']):
        x, y, z = ellipse_xyz(p, np.ones([3]) * r)
        ax.plot_wireframe(x, y, z, color=c)
    fig.show()

    return fig


def ellipse_xyz(center, extent):
    [a, b, c] = extent
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x = a * np.cos(u) * np.sin(v) + center[0]
    y = b * np.sin(u) * np.sin(v) + center[1]
    z = c * np.cos(v) + center[2]

    return x, y, z


class TestSpheres(unittest.TestCase):
    """
    """
    def test_overlaps_non_overlapping(self):
        pos1 = np.array([0, 0, 0])
        pos2 = np.array([2, 0, 0])
        pos3 = np.array([0, 0, 2])
        r1 = 1
        r2 = 1
        r3 = 1
        vol = threespheres.triple_overlap(pos1, pos2, pos3, r1, r2, r3)

        self.assertEqual(vol, 0, "Overlap found when not possible")
        return None

    def test_overlaps(self):
        p1 = np.array([0, 0, 0])
        p2 = np.array([1, 0, 0])
        p3 = np.array([0, 1, 0])

        for ii in range(100):
            r1 = 1 * np.random.random()
            r2 = 1 * np.random.random()
            r3 = 1 * np.random.random()
            vol = threespheres.triple_overlap(p1, p2, p3, r1, r2, r3)

        self.assertGreaterEqual(vol, 0, "Volume is zero or less")
        return None

    def test_research_paper_value(self):
        p1 = np.array([2. * np.cos(np.arccos(11. / 16.)),
                       2. * np.sin(np.arccos(11. / 16.)), 0])
        p2 = np.array([0, 0, 0])
        p3 = np.array([4, 0, 0])
        d12 = np.sum((p1 - p2) ** 2) ** 0.5
        d23 = np.sum((p2 - p3) ** 2) ** 0.5
        d13 = np.sum((p1 - p3) ** 2) ** 0.5
        r1 = 1
        r2 = 2
        r3 = 3
        vol = threespheres.triple_overlap(p1, p2, p3, r1, r2, r3)
        v_mc = self.mc_triple_volume(p1, p2, p3, r1, r2, r3)
        v = np.round(vol, 4)
        print("d12: {0} r1: {1}".format(d12, r1))
        print("d23: {0} r2: {1}".format(d23, r2))
        print("d13: {0} r3: {1}".format(d13, r3))
        print("Volume: {0}".format(v))
        print("MC Volume: {0}".format(v_mc))
        self.assertEqual(v, 0.5737, "Volume differs to " +
                         "paper, got {}".format(v))
        return None

    def test_mc(self):
        for ii in range(50):
            rvec = np.random.random(3)
            pvec = [np.random.random(3) for ii in range(3)]
            v_mc = self.mc_triple_volume(pvec[0], pvec[1], pvec[2],
                                         rvec[0], rvec[1], rvec[2])
            v_math = threespheres.triple_overlap(
                pvec[0], pvec[1], pvec[2], rvec[0], rvec[1], rvec[2])

            v_mc = np.round(v_mc, 1)
            v_math = np.round(v_math, 1)
            if v_mc == v_math:
                print("Correct match for v_mc: {0} and v_math: {1}".format(v_mc, v_math))
            else:
                print("MC and analytic values differ\n" +
                      "v_mc: {0} v_math: {1}".format(v_mc, v_math))

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for (r, p, c) in zip(rvec, pvec, ['r', 'g', 'b']):
                    x, y, z = ellipse_xyz(p, np.ones([3]) * r)
                    ax.plot_wireframe(x, y, z, color=c)
                fig.show()

                print("repeating with more n")
                v_mc = self.mc_triple_volume(pvec[0], pvec[1], pvec[2],
                                             rvec[0], rvec[1], rvec[2], n=1e6)
                v_mc = np.round(v_mc, 1)
                if v_mc != v_math:
                    print("MC and analytic values still differ\n" +
                          "v_mc: {0} v_math: {1}".format(v_mc, v_math))
                    print("repeating with more n")
                    v_mc = self.mc_triple_volume(pvec[0], pvec[1], pvec[2],
                                                 rvec[0], rvec[1], rvec[2],
                                                 n=1e7)

                    self.assertEqual(v_mc, v_math,
                                     "MC and analytic values differ\n" +
                                     "v_mc: {0} v_math: {1}".format(v_mc,
                                                                    v_math))
            pass

    @staticmethod
    def mc_triple_volume(p1, p2, p3, r1, r2, r3, n=1e5):
        rvec = np.array([r1, r2, r3])
        minx = min(np.array([p1[0], p2[0], p3[0]]) - rvec)
        miny = min(np.array([p1[1], p2[1], p3[1]]) - rvec)
        minz = min(np.array([p1[2], p2[2], p3[2]]) - rvec)
        maxx = max(np.array([p1[0], p2[0], p3[0]]) + rvec)
        maxy = max(np.array([p1[1], p2[1], p3[1]]) + rvec)
        maxz = max(np.array([p1[2], p2[2], p3[2]]) + rvec)

        mins = np.array([minx, miny, minz])
        maxs = np.array([maxx, maxy, maxz])
        centres = 0.5 * (maxs + mins)
        ranges = maxs - mins

        in_overlap = 0.
        in_circle = lambda p, c, r: sum((p - c) ** 2.) ** 0.5 < r
        for ii in xrange(int(n)):
            position = (np.random.random(3) - 0.5) * ranges + centres
            in1 = in_circle(position, p1, r1)
            in2 = in_circle(position, p2, r2)
            in3 = in_circle(position, p3, r3)
            if in1 and in2 and in3:
                in_overlap += 1
        vol_total = np.product(ranges)
        return vol_total * in_overlap / n


if __name__ == '__main__':
    unittest.main(verbosity=2)
