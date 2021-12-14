__all__ = ['observing_bands']

import astropy.units as u

observing_bands = {
    'U' : (365*u.nm, 66*u.nm),
    'B' : (445*u.nm, 94*u.nm),
    'G' : (464*u.nm, 128*u.nm),
    'V' : (551*u.nm, 88*u.nm),
    'R' : (658*u.nm, 138*u.nm),
    'I' : (806*u.nm, 149*u.nm),
    'Y' : (1020*u.nm, 120*u.nm),
    'J' : (1220*u.nm, 213*u.nm),
    'H' : (1630*u.nm, 307*u.nm),
    'K' : (2190*u.nm, 390*u.nm),
    'L' : (3450*u.nm, 472*u.nm),
}
