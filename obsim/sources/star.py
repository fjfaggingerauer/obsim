from .base import Source
from ..config import default_units
from ..util import make_properties

import astropy.units as u
import astropy.constants as const
import numpy as np

__all__ = ['Star']

_L0 = 3.0128E28 * u.W # https://www.iau.org/static/resolutions/IAU2015_English.pdf eq.1
_f0 = 2.518021002E-8 * u.W/u.m**2 # https://www.iau.org/static/resolutions/IAU2015_English.pdf eq.3

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
    property_list = {
        'radius' : {
            'unit' : default_units.length,
            'default' : 1*u.R_sun
        },
        'temperature' : {
            'unit' : default_units.temperature,
            'default' : 6000 * u.K
        },
        'reference_time' : {
            'unit' : default_units.time,
            'default' : 0
        },
        'current_time' : {
            'unit' : default_units.time,
            'default' : 0
        },
        'mass' : {
            'unit' : default_units.mass,
            'default' : 1*u.M_sun
        },
        'distance' : {
            'unit' : default_units.length,
            'default': 10*u.pc
        },
        'radial_velocity' : {
            'unit' : default_units.velocity,
            'default' : 0
        },
        'surface_gravity' : {
            'unit' : default_units.acceleration,
            'functions' : [(surface_gravity, ('mass', 'radius'))]
        },
        'luminosity' : {
            'unit' : default_units.power,
            'functions' : [(luminosity1, ('absolute_bolometric_magnitude',)),
                           (luminosity2, ('irradiance', 'distance')),                
                           (stefan_boltzmann_luminosity, ('temperature', 'radius'))],
            'default' : 1.0 * const.L_sun
        },
        'absolute_bolometric_magnitude': {
            'functions' : [(absolute_bolometric_magnitude, ('luminosity',))],
        },
        'irradiance': {
            'unit': default_units.power / default_units.area,
            'functions' : [(irradiance1, ('luminosity', 'distance')),
                           (irradiance2, ('apparent_bolometric_magnitude',))],
        },
        'apparent_bolometric_magnitude': {
            'functions': [(apparent_bolometric_magnitude, ('irradiance',))]
        }
    }
    def __init__(self, wavelengths=None, location=[0,0], **kwargs):
        make_properties(self, self.property_list, kwargs)
        super().__init__(wavelengths, location)

    def evolve(self, evolve_parameters):
        self.current_time += evolve_parameters['timestep']
    
    def reset(self):
        self.current_time = 0 * default_units.time