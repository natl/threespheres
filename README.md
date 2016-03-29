***threespheres module***

Routines for calculating the intersection volume of three spheres.

In threespheres.py, there are two methods for calculating the volume intersected
by three spheres. One is analytic and the second is a Monte Carlo estimation.

*Analytic (Exact) value*

This comes from a paper by Gibson and Scheraga (J. Phys. Chem., 1987). The
paper suggests the method is general to all triple overlaps, however I have
noticed that it fails for certain geometries where the centre of one circle is
significantly inside both of the other two. In cases where this may occur, it
is flagged and the default behaviour is to use Monte Carlo methods to estimate
the overlap volume, to verify whether the analytic value is valid.

Usage:

```
from threespheres import triple_overlap

p1 = np.array([2. * np.cos(np.arccos(11. / 16.)),
               2. * np.sin(np.arccos(11. / 16.)), 0])
p2 = np.array([0, 0, 0])
p3 = np.array([4, 0, 0])

r1 = 1
r2 = 2
r3 = 3

print triple_overlap(p1, p2, p3, r1, r2, r3)
```
