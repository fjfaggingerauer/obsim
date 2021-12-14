__all__ = ['Telescope',]

import astropy.units as u
import astropy.coordinates as c
import astropy.time as t
import hcipy as hp

from ..config import default_units, simulation_units
from ..util import make_properties

# Conversion functions for grid properties
# pupil grid: diameter = pupil_grid_size * pupil_grid_resolution
def get_diameter(pupil_grid_size, pupil_grid_resolution):
    return pupil_grid_resolution * pupil_grid_size
def get_pupil_grid_size(diameter, pupil_grid_resolution):
    return ((diameter/pupil_grid_resolution).value).astype(int)
def get_pupil_grid_resolution(diameter, pupil_grid_size):
    return diameter/pupil_grid_size

# focal grid
# q = focal_grid_size / num_airy
# field_of_view = focal_grid_size * focal_grid_resolution / focal_length
# phys_scale = f_number * reference_wavelength
# field_of_view = num_airy * reference_wavelength/diameter
# f_number = focal_length / diameter
# phys_scale = q * focal_grid_resolution
def get_q1(focal_grid_size, num_airy):
    return focal_grid_size/num_airy
def get_focal_grid_size1(q, num_airy):
    return q * num_airy
def get_num_airy1(q, focal_grid_size):
    return focal_grid_size / q
def get_field_of_view1(focal_grid_size, focal_grid_resolution, focal_length):
    return focal_grid_size * focal_grid_resolution / focal_length
def get_focal_grid_size2(field_of_view, focal_grid_resolution, focal_length):
    return ((field_of_view.to(u.rad) * focal_length/focal_grid_resolution).value).astype(int)
def get_focal_grid_resolution1(field_of_view, focal_grid_size, focal_length):
    return field_of_view.to(u.rad).value/focal_grid_size * focal_length
def get_focal_length1(focal_grid_size, focal_grid_resolution, field_of_view):
    return focal_grid_size * focal_grid_resolution / field_of_view.to(u.rad).value
def get_phys_scale1(f_number, reference_wavelength):
    return f_number * reference_wavelength
def get_f_number1(phys_scale, reference_wavelength):
    return (phys_scale / reference_wavelength).value
def get_reference_wavelength1(phys_scale, f_number):
    return phys_scale / f_number
def get_field_of_view2(num_airy, reference_wavelength, diameter):
    return (num_airy * reference_wavelength / diameter) * u.rad
def get_num_airy2(field_of_view, reference_wavelength, diameter):
    return field_of_view.to(u.rad).value * (diameter/reference_wavelength).value
def get_reference_wavelength2(field_of_view, num_airy, diameter):
    return field_of_view.to(u.rad).value / num_airy * diameter
# no function for diameter, this should only be determined by the pupil grid
def get_f_number2(focal_length, diameter):
    return (focal_length/diameter).value
def get_focal_length2(f_number, diameter):
    return f_number * diameter
def get_phys_scale2(q, focal_grid_resolution):
    return q * focal_grid_resolution
def get_q2(phys_scale, focal_grid_resolution):
    return (phys_scale / focal_grid_resolution).value
def get_focal_grid_resolution2(phys_scale, q):
    return phys_scale / q

class Telescope(object):
    property_list = {
        # pupil grid
        'diameter' : { # telescope primary mirror diameter
            'unit': default_units.length,
            'default' : 1*u.m,
            'functions' : [(get_diameter, ("pupil_grid_size", "pupil_grid_resolution"))]
        },
        'pupil_grid_size' : { # number of pixels in telescope pupil grid (for hcipy)
            'default' : 128,
            'functions' : [(get_pupil_grid_size, ("diameter", "pupil_grid_resolution"))]
        },
        'pupil_grid_resolution' : { # physical size of a pixel in the pupil grid
            'unit' : default_units.length,
            'functions' : [(get_pupil_grid_resolution, ("diameter", "pupil_grid_size"))]
        },
        # focal grid
        'field_of_view' : { # field of view of the telescope
            'unit' : default_units.angle,
            'functions' : [(get_field_of_view1, ("focal_grid_size", "focal_grid_resolution", "focal_length")),
                           (get_field_of_view2, ("num_airy", "reference_wavelength", "diameter"))]
        },
        'reference_wavelength' : { # reference wavelength in lambda/d calculations
            'unit' : default_units.length,
            'default' : 1E-6 * u.m,
            'functions' : [(get_reference_wavelength1, ("phys_scale", "f_number")),
                           (get_reference_wavelength2, ("field_of_view", "num_airy", "diameter"))]
        },
        'f_number': { # telescope primary f-number
            'default' : 1,
            'functions' : [(get_f_number1, ("phys_scale", "reference_wavelength")),
                           (get_f_number2, ("focal_length", "diameter"))]
        },
        'q' : { # number of pixels per lambda/d in focal plane
            'default' : 2,
            'functions' : [(get_q1, ("focal_grid_size", "num_airy")),
                           (get_q2, ("phys_scale", "focal_grid_resolution"))]
        },
        'phys_scale' : { # physical size of a lambda/d distance in the focal plane
            'unit' : default_units.length,
            'functions' : [(get_phys_scale1, ("f_number", "reference_wavelength")),
                           (get_phys_scale2, ("q", "focal_grid_resolution"))]
        },
        'focal_grid_size' : { # number of pixels in focal plane grid (for hcipy)
            'functions' : [(get_focal_grid_size1, ("q", "num_airy")),
                           (get_focal_grid_size2, ("field_of_view", "focal_grid_resolution", "focal_length"))]
        },
        'focal_length' : { # effective focal length of primary
            'unit' : default_units.length,
            'functions' : [(get_focal_length1, ("focal_grid_size", "focal_grid_resolution", "field_of_view")),
                           (get_focal_length2, ("f_number", "diameter"))]
        },
        'num_airy' : { # number of airy rings in the focal grid (for hcipy)
            'functions' : [(get_num_airy1, ("q", "focal_grid_size")),
                           (get_num_airy2, ("field_of_view", "reference_wavelength", "diameter"))]
        },
        'focal_grid_resolution' : { # size of a pixel in focal plane grid (for hcipy)
            'unit' : default_units.length,
            'functions' : [(get_focal_grid_resolution1, ("field_of_view", "focal_grid_size", "focal_length")),
                           (get_focal_grid_resolution2, ("phys_scale", "q"))]
        },
        # aperture
        'aperture_type' : { # shape of aperture (including spiders etc.) (for hcipy aperture generator)
            'type' : str,
            'default' : 'circular'
        },
        'aperture_generator' : { # hcipy aperture generator
        },
        # pointing of the telescope
        'pointing' : { # direction the telescope is pointed when sources don't have a physical location
            'unit': default_units.angle,
            'default' : [0,0]
        },
        'physical_pointing' : { # Altitude/azimuth of telescope pointing
            'type' : c.SkyCoord,
        },
        # properties for sources with physical location
        'rotation' : { # TBD
            'unit' : default_units.angle,
            'default' : 0 * u.rad
        },
        'location': { # location of the telescope on Earth
            'type' : c.EarthLocation,
        },
        'observation_time': { # date & time of observation
            'type' : t.Time,
        },
        'reference_azimuth': { # TBD
            'unit' : default_units.angle,
            'default' : 0 * u.rad
        },
        'altitude_limits':{ # minimum/maximum altitude that can be pointed at
            'unit' : default_units.angle,
            'default' : [0, 90] * u.deg,
        },
    }

    def __init__(self, **kwargs):
        make_properties(self, self.property_list, kwargs)
        self._tracked_target = None
    
    @property
    def pupil_grid(self):
        return hp.make_pupil_grid(self.pupil_grid_size, self.diameter.to(simulation_units.length).value)
    
    @property
    def focal_grid(self):
        return hp.make_focal_grid(self.q, self.num_airy, self.phys_scale.to(simulation_units.length).value)
    
    @property
    def aperture_function(self):
        if self.aperture_generator is not None:
            return self.aperture_generator
        elif self.aperture_type == 'circular':
            return hp.circular_aperture(self.diameter.to(simulation_units.length).value)
        else:
            raise NotImplementedError(f"Aperture type '{self.aperture_type}' not supported.")
    
    def aperture(self, component_type='hcipy'):
        if component_type == 'hcipy':
            from ..components import Apodizer
            return Apodizer(self.aperture_function, 'pupil')
        else:
            raise ValueError(f"Unrecognized type '{component_type}' for telescope apodizer.")
    
    def propagator(self, component_type='hcipy'):
        if component_type == 'hcipy':
            from ..components import FraunhoferPropagator
            return FraunhoferPropagator(propagator = hp.FraunhoferPropagator(self.pupil_grid, self.focal_grid, self.focal_length.to(simulation_units.length).value))
        else:
            raise ValueError(f"Unrecognized type '{component_type}' for telescope propagator.")

    def point_at(self, source):
        def point(loc):
            if self.location is None:
                raise ValueError("When a source has a physical location, the telescope must have this as well.")
            if self.observation_time is None:
                raise ValueError("When a source has a physical location, an observation time must be given.")
            
            self.physical_pointing = loc.transform_to(c.AltAz(obstime = self.observation_time, location = self.location))

            if self.physical_pointing.alt < self.altitude_limits.min() or self.physical_pointing.alt > self.altitude_limits.max():
                raise RuntimeError(f"Cannot point telescope to an altitude of {self.physical_pointing.alt}, it must be in the range {self.altitude_limits}.")

        if isinstance(source, c.SkyCoord):
            point(source)
        
        if source.physical_location is None:
            self.pointing = source.location
        else:
            point(source.physical_location)                
    
    @property
    def tracked_target(self):
        return self._tracked_target

    @property
    def tracking_target(self):
        return self.tracked_target is not None
    

    def track(self, source):
        if source.physical_location is not None:
            self._tracked_target = source
        
        self.point_at(source)
    
    def __call__(self, val):
        ap = self.aperture()(val)
        prop = self.propagator()(ap)

        return prop
    
    def to_dict(self):
        tree = {'properties': self.properties.to_dict(),
                #'tracked_target': self.tracked_target, # TBD
                #'pointing' : self.pointing} # TBD
        }
    
    @classmethod
    def from_dict(cls, tree):
        kwargs = {key: tree['properties']['values'][key] for key in tree['properties']['values'] if tree['properties']['externally_set'][key]}
        obj = cls(**kwargs)

        #if tree['tracked_target'] is not None:
        #    obj.track()
        #if tree['pointing'] is not None:
        #    obj.point_at()