from .base import Source
from ..config import default_units
from ..util import make_properties

import astropy.units as u
import astropy.constants as const
import numpy as np

__all__ = ['Star']

_L0 = 3.0128E28 * u.W  # https://www.iau.org/static/resolutions/IAU2015_English.pdf eq.1
_f0 = 2.518021002E-8 * u.W/u.m**2  # https://www.iau.org/static/resolutions/IAU2015_English.pdf eq.3

# --- Auxiliary functions ---
def surface_gravity(mass, radius):
    return const.G*mass/radius**2

def stefan_boltzmann_luminosity(temperature, radius):
    return const.sigma_sb * temperature**4 * radius**2 * np.pi

def absolute_bolometric_magnitude(luminosity):
    return -2.5 * np.log10((luminosity/_L0).value)

def irradiance1(luminosity, distance):
    return luminosity / (4*np.pi*distance**2)

def irradiance2(apparent_bolometric_magnitude):
    return _f0 * 10**(-0.4 * apparent_bolometric_magnitude)

def apparent_bolometric_magnitude(irradiance):
    return -2.5 * np.log10((irradiance/_f0).value)

def luminosity1(absolute_bolometric_magnitude):
    return _L0 * 10**(-0.4 * absolute_bolometric_magnitude)

def luminosity2(irradiance, distance):
    return 4*np.pi*distance**2 * irradiance


class Star(Source):
    '''
    Class for simulating a Star as input Source.

    Parameters
    ----------
    wavelengths: UnitField
        Wavelengths at which the simulation is done and the spectral model
        should be evaluated.
    location: astropy.SkyCoord or astropy.units.Quantity
        Location of the star in sky coordinates or w.r.t. the telescope
        pointing (e.g. (0, 0) puts it in the center of the field of view).

    PropertyList Parameters
    -----------------------
    radius: float or astropy.units.Quantity
        Radius of the star in solar radii or with specified units
    temperature: float or astropy.units.Quantity
        Temperature of the star in K or with specified units
    mass: float or astropy.units.Quanity
        Mass of the star in solar mass or with specified units
    distance: float or astropy.units.Quantity
        Distance of the star in parsec or with specified units
    radial_velocity: float or astropy.unit.Quantity
        Radial velocity of the star w.r.t. the observer in km/s or with
        specified units
    surface gravity: float or astropy.unit.Quantity
        Surface gravity of the star in cm/s^2 or with specified units
    luminosity: float or astropy.unit.Quantity
        Luminosity of the star in solar luminosity or with specified units
    absolute_bolometric_magnitude: float
        Absolute bolometric magnitude of the star
    apparent_bolometric_magnitude: float
        Apparent bolometric magnitude of the star
    irradiance: float or astropy.unit.Quantity
        Irradiance of the star in W/m^2 or with specified units
    '''
    property_list = {
        'radius': {
            'unit': u.R_sun,
            'default': 1*u.R_sun
        },
        'temperature': {
            'unit': default_units.temperature,
            'default': 6000 * u.K
        },
        'mass': {
            'unit': u.M_sun,
            'default': 1*u.M_sun
        },
        'distance': {
            'unit': u.pc,
            'default': 10*u.pc
        },
        'radial_velocity': {
            'unit': u.km/u.s,
            'default': 0
        },
        'surface_gravity': {
            'unit': u.cm/u.s**2,
            'functions': [(surface_gravity, ('mass', 'radius'))]
        },
        'luminosity': {
            'unit': u.L_sun,
            'functions': [(luminosity1, ('absolute_bolometric_magnitude',)),
                          (luminosity2, ('irradiance', 'distance')),
                          (stefan_boltzmann_luminosity, ('temperature',
                                                         'radius'))],
            'default': 1.0 * const.L_sun
        },
        'absolute_bolometric_magnitude': {
            'functions': [(absolute_bolometric_magnitude, ('luminosity',))],
        },
        'irradiance': {
            'unit': default_units.power / default_units.area,
            'functions': [(irradiance1, ('luminosity', 'distance')),
                          (irradiance2, ('apparent_bolometric_magnitude',))],
        },
        'apparent_bolometric_magnitude': {
            'functions': [(apparent_bolometric_magnitude, ('irradiance',))]
        }
    }

    def __init__(self, wavelengths=None, location=[0, 0], **kwargs):
        make_properties(self, self.property_list, kwargs)
        super().__init__(wavelengths, location)
