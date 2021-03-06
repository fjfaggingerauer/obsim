import numpy as np
import astropy.units as u

from .units import set_units

__all__ = ['calculate_airmass']


@set_units(zenith_angle=u.rad)
def calculate_airmass(zenith_angle, method='simple'):
    '''
    Calculate the airmass for a given zenith angle using some common methods
    for the airmass.

    Parameters
    ----------
    zenith_angle : astropy.Quantity
        Angle(s) for which to calculate the airmass. Must have units of degrees
        or radians (or equivalent).
    method : {"simple", "Hardie", "Rozenberg", "Kasten&Young", "Pickering"}
        Method for calculating the airmass. Currently supported methods are
        from [1-4] (in order). Simple evaluates the airmass as
        sec(zenith_angle).

    Returns
    -------
    airmass : array
        Airmass per zenith angle. Unitless.

    [1] : https://ui.adsabs.harvard.edu/abs/1962aste.book.....H/abstract
    [2] : Rozenberg, G. V. 1966. Twilight: A Study in Atmospheric Optics.
          New York: Plenum Press, 160.
    [3] : https://ui.adsabs.harvard.edu/abs/1989ApOpt..28.4735K/abstract
    [4] : http://www.dioi.org/vols/wc0.pdf
    '''

    c = np.cos(zenith_angle)
    if method == 'simple':
        return 1/c
    # https://ui.adsabs.harvard.edu/abs/1967AJ.....72..945Y/abstract
    # elif method == 'Young&Irvine':
    #    return 1/c * (1-0.0012*(1/c**2 -1))
    elif method == 'Hardie':
        return 1/c - 0.0018167*(1/c-1) - 0.002875*(1/c-1)**2 \
               - 0.0008083*(1/c-1)**3
    elif method == 'Rozenberg':
        return (c+0.025*np.exp(-11*c))**(-1)
    elif method == 'Kasten&Young':
        return (c + 0.50572*(96.07995*u.deg
                - zenith_angle.to(u.deg))**(-1.6364))**(-1)
    # https://ui.adsabs.harvard.edu/abs/1994ApOpt..33.1108Y/abstract
    # elif method == 'Young':
    #    return (1.002432*c**2 + 0.148386*c + 0.0096467)/(c**3
    #               + 0.149864*c**2 + 0.0102963*c + 0.000303978)
    elif method == 'Pickering':
        h = 90*u.deg - zenith_angle
        return np.sin(h+244/(165+47*h**(1.1)))**(-1)
    else:
        raise ValueError(f"Method '{method}' not supported.")
