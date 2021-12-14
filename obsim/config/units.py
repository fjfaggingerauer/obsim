__all__ = ['default_units', 'simulation_units']

import astropy.units as u
import hcipy as hp

class default_units:
    length = u.m
    time = u.s
    mass = u.kg
    temperature = u.K
    angle = u.rad

    area = length**2
    volume = length**3
    velocity = length/time
    acceleration = velocity/time
    frequency = time**(-1)
    
    angular_area = angle**2

    #energy = mass * velocity**2
    energy = u.J
    power = energy/time
    flux = power/area
    flux_wavelength_density = flux/length
    flux_frequency_density = flux/frequency
    flux_wavelength_angular_density = flux_wavelength_density/angular_area
    flux_frequency_angular_density = flux_frequency_density/angular_area

class simulation_units:
    length = u.m
    time = u.s
    mass = u.kg
    temperature = u.K
    angle = u.rad

    area = length**2
    volume = length**3
    velocity = length/time
    frequency = time**(-1)
    
    angular_area = angle**2

    #energy = mass * velocity**2
    energy = u.J
    power = energy/time
    flux = power/area
    flux_wavelength_density = flux/length
    flux_frequency_density = flux/frequency
    flux_wavelength_angular_density = flux_wavelength_density/angular_area
    flux_frequency_angular_density = flux_frequency_density/angular_area