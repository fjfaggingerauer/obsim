import pathlib
import sys

p = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(p))

import obsim as obs
import astropy.units as u


def make_test_obj_unitless():
    def add(x, y):
        return x+y

    class A:
        property_list = {
            'x': {'type': float,
                  'default': 1.0},
            'y': {'type': int,
                  'default': 5},
            'z': {'type': float,
                  'functions': ((add, ('x', 'y')),)}
        }

        def __init__(self, **kwargs):
            obs.make_properties(self, self.property_list, kwargs)
    return A()


def make_test_obj_units():
    def add(x, y):
        return x+y

    def mul(x, y):
        return x*y

    class B:
        property_list = {
            'x': {'unit': u.s,
                  'default': 1.0},
            'y': {'unit': u.m,
                  'default': 5},
            'z': {'unit': u.s,
                  'functions': ((add, ('x', 'y')),)},
            'w': {'unit': u.s*u.m,
                  'functions': ((mul, ('x', 'y')),)}
        }

        def __init__(self, **kwargs):
            obs.make_properties(self, self.property_list, kwargs)
    return B()


def test_property_initialization():
    A1 = make_test_obj_unitless()
    A2 = make_test_obj_units()

    x = A1.x
    y = A1.y
    z = A1.z
    x = A2.x
    y = A2.y
    w = A2.w


def test_property_default():
    A1 = make_test_obj_unitless()
    A2 = make_test_obj_units()

    assert(A1.x == 1.0)
    assert(A1.y == 5)
    assert(A1.z == 6.0)
    assert(A2.x == 1*u.s)
    assert(A2.y == 5*u.m)
    assert(A2.w == 5*u.s*u.m)


def test_property_setting():
    A1 = make_test_obj_unitless()
    A2 = make_test_obj_units()

    A1.x = 2.0
    A1.y = 8
    A2.x = 2*u.s
    A2.y = 3*u.m

    try:
        A2.x = 5*u.m
        assert(False)
    except u.UnitsError:
        pass

    try:
        A1.y = 5.0
        assert(False)
    except TypeError:
        pass

    assert(A1.x == 2.0)
    assert(A1.y == 8)
    assert(A1.z == 10.0)
    assert(A2.x == 2*u.s)
    assert(A2.y == 3*u.m)
    assert(A2.w == 6*u.s*u.m)


if __name__ == '__main__':
    test_property_initialization()
    test_property_default()
    test_property_setting()
