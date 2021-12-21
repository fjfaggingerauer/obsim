from .base import Source
from ..config import default_units, simulation_units
from ..util import make_properties
from .star import Star

import astropy.units as u
import astropy.constants as const
import numpy as np

__all__ = ['Planet']

def surface_gravity(mass, radius):
    return const.G*mass/radius**2

def keplerian_period(M_star, a):
    return np.sqrt((4*np.pi**2)/(const.G * M_star) * a**3)

def relative_angular_position(a, phase, distance, inclination):
    return a/distance * np.array([np.sin(phase), np.cos(phase)*np.cos(inclination)]) * u.rad

def orbital_phase(t, t0, period):
    return 2*np.pi*((t-t0)/period % 1) * u.rad

def id(x):
    return x

def total_position(relative_position, position):
    return relative_position + position

def relative_radial_velocity(orbital_period, sma, orbital_phase, inclination):
    return (-2*np.pi*sma/orbital_period) * np.sin(orbital_phase) * np.sin(inclination)

def radial_velocity(v_sys, relative_radial_velocity):
    return v_sys + relative_radial_velocity


class Planet(Source):
    '''
    Class for simulating a Planet as input Source.

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
        Radius of the star in Jupiter radii or with specified units
    temperature: float or astropy.units.Quantity
        Temperature of the star in K or with specified units
    mass: float or astropy.units.Quanity
        Mass of the star in Jupiter mass or with specified units
    surface gravity: float or astropy.unit.Quantity
        Surface gravity of the star in cm/s^2 or with specified units
    sma: float or astropy.units.Quantity
        Semi-major axis of the planet orbit in AU or with specified units
    inclination: float or astropy.units.Quantity
        Inclination of the planet orbit in degree or with specified units
    host: obsim.sources.Star
        Host star of the planet
    distance: float or astropy.units.Quantity
        Distance of the star in parsec or with specified units
    radial_velocity: float or astropy.unitss.Quantity
        Radial velocity of the host planet w.r.t. the observer in km/s or with
        specified units
    relative_radial_velocity: float or astropy.unit.Quantity
        Radial velocity of the host planet w.r.t. the host star in km/s or with
        specified units
    reference_time: float or astropy.units.Quantity
        Time offset for the planet orbit
    current_time: float or astropy.units.Quantity
        Current time used to calculate the planet's position
    orbital_phase: float or astropy.units.Quantity
        Orbital phase of the planet's orbit in radians
    orbital_period: float or astropy.units.Quantity
        Orbital period of the planet in years
    '''
    property_list = {
        'radius': {
            'unit': u.R_jup,
        },
        'temperature': {
            'unit': default_units.temperature,
        },
        'mass': {
            'unit': u.M_jup,
        },
        'surface_gravity': {
            'unit': u.cm/u.s**2,
            'functions': [(surface_gravity, ('mass', 'radius'))]
        },
        'sma': {
            'unit': u.AU,
        },
        'inclination': {
            'unit': u.deg,
            'default': 0*u.deg,
        },
        'host': {
            'type': Star,
        },
        'distance': {
            'unit': u.pc,
            'functions': [(id, ("host.distance",))]
        },
        'reference_time': {
            'unit': default_units.time,
            'default': 0
        },
        'current_time': {
            'unit': default_units.time,
            'default': 0
        },
        'orbital_phase': {
            'unit': default_units.angle,
            'functions': [(orbital_phase, ("current_time", "reference_time",
                                           "orbital_period"))]
        },
        'relative_location': {
            'unit': default_units.angle,
            'functions': [(relative_angular_position, ('sma', 'orbital_phase',
                                                       'distance',
                                                       'inclination'))]
        },
        'orbital_period': {
            'unit': u.yr,
            'functions': [(keplerian_period, ("host.mass", "sma"))]
        },
        'relative_radial_velocity': {
            'unit': u.km/u.s,
            'functions': [(relative_radial_velocity, ('orbital_period', 'sma',
                                                      'orbital_phase',
                                                      'inclination'))],
        },
        'radial_velocity': {
            'unit': u.km/u.s,
            'functions': [(radial_velocity, ('host.radial_velocity',
                                             'relative_radial_velocity'))],
        },
        '_location': {
            'unit': default_units.angle,
            'functions': [(total_position, ("relative_location",
                                            "host.location"))]
        }
    }

    def __init__(self, wavelengths=None, location=None, **kwargs):
        make_properties(self, self.property_list, kwargs)
        super().__init__(wavelengths, location)

    def evolve(self, evolve_parameters):
        if self.properties.orbital_phase.is_set:
            # convert orbital phase to reference time so updating time is not
            # overwritten by constant orbital phase
            p = self.orbital_phase
            self.orbital_phase = None
            self.reference_time = -p.to(u.rad).value/(2*np.pi) * \
                self.orbital_period

        self.current_time += evolve_parameters['timestep']

    def reset(self):
        self.current_time = 0 * simulation_units.time
