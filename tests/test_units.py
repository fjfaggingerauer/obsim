import pathlib
import sys

p = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(p))

import obsim as obs
import astropy.units as u


def test_set_units_basic():
    @obs.set_units(x=u.m, y=u.s*u.m)
    def f(x, y):
        assert(x.unit == u.m)
        assert(y.unit == u.s*u.m)

    f(3*u.m, 2*u.s*u.m)

    try:
        f(3*u.s, 2*u.s*u.m)
        assert(False)
    except u.UnitsError:
        pass


def test_strip_units_basic():
    @obs.strip_units(x=u.m, y=u.s*u.m)
    def g(x, y):
        assert(not hasattr(x, 'unit'))
        assert(not hasattr(y, 'unit'))

    g(3*u.m, 2*u.s*u.m)

    try:
        g(3*u.s, 2*u.s*u.m)
        assert(False)
    except u.UnitsError:
        pass


def test_none_behaviour():
    @obs.set_units(x=u.m)
    def f(x=None):
        assert(x is None)

    @obs.strip_units(x=u.m)
    def g(x=None):
        assert(x is None)

    @obs.set_units(x=u.m)
    def h(x):
        assert(x is None)

    @obs.set_units(x=u.m)
    def i(x):
        assert(x is None)

    f(None)
    g(None)

    try:
        h(None)
        assert(False)
    except ValueError:
        pass

    try:
        i(None)
        assert(False)
    except ValueError:
        pass


if __name__ == '__main__':
    test_set_units_basic()
    test_strip_units_basic()
    test_none_behaviour()
