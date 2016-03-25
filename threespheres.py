from __future__ import division, unicode_literals, print_function
import numpy as np


def double_overlap(pos1, pos2, r1, r2):
    """
    double_overlap(pos1, pos2, r1, r2)

    Calculate the overlap volume of two spheres of radius r1, r2, at positions
        pos1, pos2
    """
    d = sum((pos1 - pos2) ** 2) ** 0.5
    # check they overlap
    if d >= (r1 + r2):
        return 0
    # check if one entirely holds the other
    if r1 > (d + r2):  # 2 is entirely contained in one
        return 4. / 3. * np.pi * r2 ** 3
    if r2 > (d + r1):  # 1 is entirely contained in one
        return 4. / 3. * np.pi * r1 ** 3

    vol = (np.pi * (r1 + r2 - d) ** 2 * (d ** 2 + (2 * d * r1 - 3 * r1 ** 2 +
                                                   2 * d * r2 - 3 * r2 ** 2)
                                         + 6 * r1 * r2)) / (12 * d)
    return vol


def get_p_values(a, b, c, alpha, beta, gamma):
    """
    Helper function for triple_overlap
    """
    t2 = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
    tabg2 = ((a + beta + gamma) * (-a + beta + gamma) *
             (a - beta + gamma) * (a + beta - gamma))

    t = t2 ** 0.5
    tabg = tabg2 ** 0.5

    a2 = a ** 2
    b2 = b ** 2
    c2 = c ** 2

    alpha2 = alpha ** 2
    beta2 = beta ** 2
    gamma2 = gamma ** 2

    p1 = ((b2 - c2 + beta2 - gamma2) ** 2 + (t - tabg) ** 2) / (4 * a2) - alpha2  # NOQA
    p2 = ((b2 - c2 + beta2 - gamma2) ** 2 + (t + tabg) ** 2) / (4 * a2) - alpha2  # NOQA

    return p1, p2


def atanpi(val):
    theta = np.arctan(val)
    if theta < 0:
        theta += np.pi
    return theta


def triple_overlap(pos1, pos2, pos3, r1, r2, r3):
    """
    triple_overlap(pos1, pos2, pos3, r1, r2, r3)

    Calculate volume overlapped by 3 spheres
    From Gibson and Scheraga (1987)
    """
    a = sum((pos3 - pos2) ** 2) ** 0.5
    b = sum((pos3 - pos1) ** 2) ** 0.5
    c = sum((pos2 - pos1) ** 2) ** 0.5
    if not ((a <= (r3 + r2)) and (b <= (r3 + r1)) and (c <= (r2 + r1))):
        return 0

    # Check if one sphere entirely contains another
    if (r1 > (b + r3)) or (r1 > (c + r2)):  # Circle 1 encloses circle 2/3
        vol = double_overlap(pos2, pos3, r2, r3)
        return vol
    elif (r2 > (a + r3)) or (r2 > (c + r1)):  # Circle 2 encloses circle 1/3
        vol = double_overlap(pos1, pos3, r1, r3)
        return vol
    elif (r3 > (b + r1)) or (r3 > (a + r2)):  # Circle 3 encloses circle 1/2
        vol = double_overlap(pos1, pos2, r1, r2)
        return vol

    alpha = r1
    beta = r2
    gamma = r3

    a2 = a ** 2
    b2 = b ** 2
    c2 = c ** 2

    alpha2 = alpha ** 2
    beta2 = beta ** 2
    gamma2 = gamma ** 2

    eps1 = (beta2 - gamma2) / a2
    eps2 = (gamma2 - alpha2) / b2
    eps3 = (alpha2 - beta2) / c2

    w2 = ((alpha2 * a2 + beta2 * b2 + gamma2 * c2) * (a2 + b2 + c2) -
          2 * (alpha2 * a2 ** 2 + beta2 * b2 ** 2 + gamma2 * c2 ** 2) +
          (a2 * b2 * c2) * (eps1 * eps2 + eps2 * eps3 + eps3 * eps1 - 1))

    if w2 > 0:
        w = w2 ** 0.5
        q1 = a * (b2 + c2 - a2 + beta2  + gamma2 - 2. * alpha2 + eps1 * (b2 - c2))  # NOQA
        q2 = b * (c2 + a2 - b2 + gamma2 + alpha2 - 2. * beta2  + eps2 * (c2 - a2))  # NOQA
        q3 = c * (a2 + b2 - c2 + alpha2 + beta2  - 2. * gamma2 + eps3 * (a2 - b2))  # NOQA

        alpha3 = alpha ** 3.
        beta3 = beta ** 3.
        gamma3 = gamma ** 3.
        aw = a * w
        bw = b * w
        cw = c * w

        vol = (w / 6. - a / 2. * (beta2  + gamma2 - a2 * (1. / 6. - eps1 ** 2 / 2.)) * atanpi(2 * w / q1)
                      - b / 2. * (gamma2 + alpha2 - b2 * (1. / 6. - eps2 ** 2 / 2.)) * atanpi(2 * w / q2)
                      - c / 2. * (alpha2 + beta2  - c2 * (1. / 6. - eps3 ** 2 / 2.)) * atanpi(2 * w / q3)
                      + (2. / 3.) * alpha3 * (atanpi(bw / (alpha * q2) * (1 - eps2)) + atanpi(cw / (alpha * q3) * (1 + eps3)))
                      + (2. / 3.) * beta3  * (atanpi(cw / (beta  * q3) * (1 - eps3)) + atanpi(aw / (beta  * q1) * (1 + eps1)))
                      + (2. / 3.) * gamma3 * (atanpi(aw / (gamma * q1) * (1 - eps1)) + atanpi(bw / (gamma * q2) * (1 + eps2))))  # NOQA

    elif (w2 < 0):
        p1, p2 = get_p_values(a, b, c, alpha, beta, gamma)
        p3, p4 = get_p_values(b, c, a, beta, gamma, alpha)
        p5, p6 = get_p_values(c, a, b, gamma, alpha, beta)

        if (p3 > 0) and (p5 > 0):
            if (p1 > 0):
                vol = 0
            if (p1 < 0):
                vol = double_overlap(pos2, pos3, r2, r3)
        elif (p1 > 0) and (p5 > 0):  # fill out...
            if (p3 > 0):
                vol = 0
            if (p3 < 0):
                vol = double_overlap(pos1, pos3, r1, r3)
        elif (p1 > 0) and (p3 > 0):
            if (p5 > 0):
                vol = 0
            if (p5 < 0):
                vol = double_overlap(pos1, pos2, r1, r2)
        elif (p1 > 0) and (p3 < 0) and (p5 < 0):  # NOQA
            vol = (double_overlap(pos1, pos2, r1, r2) +
                   double_overlap(pos1, pos3, r1, r3) -
                   4. / 3. * np.pi * r1 ** 3.)
        elif (p1 < 0) and (p3 > 0) and (p5 < 0):  # NOQA
            vol = (double_overlap(pos1, pos2, r1, r2) +
                   double_overlap(pos2, pos3, r2, r3) -
                   4. / 3. * np.pi * r2 ** 3.)
        elif (p1 < 0) and (p3 < 0) and (p5 > 0):  # NOQA
            vol = (double_overlap(pos1, pos3, r1, r3) +
                   double_overlap(pos2, pos3, r2, r3) -
                   4. / 3. * np.pi * r3 ** 3.)
        else:
            print("Unknown case???")
            vol = np.nan
    else:
        vol = 0

    return vol
