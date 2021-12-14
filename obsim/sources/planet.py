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
    property_list = {
        'radius' : {
            'unit' : default_units.length,
        },
        'temperature' : {
            'unit' : default_units.temperature,
        },
        'reference_time' : {
            'unit' : default_units.time,
            'default' : 0
        },
        'current_time' : {
            'unit' : default_units.time,
            'default' : 0
        },
        'sma' : { # semi-major axis/average separation
            'unit' : default_units.length,
        },
        'inclination' : {
            'unit' : default_units.angle,
            'default' : 0*u.deg,
        },
        'mass' : {
            'unit' : default_units.mass,
        },
        'host' : {
            'type' : Star,
        },
        'distance' : {
            'unit' : default_units.length,
            'functions' : [(id, ("host.distance",))]
        },
        'orbital_phase' : {
            'unit' : default_units.angle,
            'functions' : [(orbital_phase, ("current_time", "reference_time", "orbital_period"))]
        },
        'relative_location' : {
            'unit' : default_units.angle,
            'functions' : [(relative_angular_position, ('sma', 'orbital_phase', 'distance', 'inclination'))]
        },
        'orbital_period' : {
            'unit' : default_units.time,
            'functions' : [(keplerian_period, ("host.mass", "sma"))]
        },
        'relative_radial_velocity' : {
            'unit' : default_units.velocity,
            'functions' : [(relative_radial_velocity, ('orbital_period', 'sma', 'orbital_phase', 'inclination'))],
        },
        'radial_velocity' : {
            'unit' : default_units.velocity,
            'functions' : [(radial_velocity, ('host.radial_velocity', 'relative_radial_velocity'))],
        },
        'surface_gravity' : {
            'unit' : default_units.length/default_units.time**2,
            'functions' : [(surface_gravity, ('mass', 'radius'))]
        },
        '_location' : {
            'unit' : default_units.angle,
            'functions' : [(total_position, ("relative_location", "host.location"))]
        }
    }
    def __init__(self, wavelengths=None, location=None, **kwargs):
        make_properties(self, self.property_list, kwargs)
        super().__init__(wavelengths, location)
    
    def evolve(self, evolve_parameters):
        if self.properties.orbital_phase.is_set:
            # convert orbital phase to reference time so updating time is not overwritten by constant orbital phase
            p = self.orbital_phase
            self.orbital_phase = None
            self.reference_time = -p.to(u.rad).value/(2*np.pi) * self.orbital_period
        
        self.current_time += evolve_parameters['timestep']
    
    def reset(self):
        self.current_time = 0 * simulation_units.time