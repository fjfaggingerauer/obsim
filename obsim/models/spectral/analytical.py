from .base import SpectralModel
from .util import make_spectrum_unit_field
from ...config import default_units
from ...util import set_units, make_properties
from ...field import Grid, UnstructuredCoords

from astropy.modeling.physical_models import BlackBody as astropy_blackbody
import numpy as np
import astropy.units as u

__all__ = ['FlatSpectrum', 'BlackBodySpectrum']

class FlatSpectrum(SpectralModel):
    @set_units(value = [default_units.flux_wavelength_density, 
                           default_units.flux_frequency_density])
    def __init__(self, value=1.):
        self.value = value
    
    @set_units(wavelengths = default_units.length)
    def at(self, wavelengths):
        grid = Grid(UnstructuredCoords(wavelengths))
        return self.value * grid.ones()


def get_temperature(host):
    return host.temperature
def get_solid_angle(host):
    return np.pi * (host.radius/host.distance)**2 * u.sr
class BlackBodySpectrum(SpectralModel):
    property_list = {
        'temperature' : {
            'unit' : default_units.temperature,
            'function' : [(get_temperature, ('source',))],
            'default' : 6000 * u.K
        },
        'solid_angle' : {
            'unit' : default_units.angular_area,
            'function' : [(get_solid_angle, ('source',))],
            'default' : np.pi * (u.Rsun/u.pc)**2 * u.sr
        },
        'source' : {}
    }
    def __init__(self, **kwargs):
        make_properties(self, self.property_list, kwargs)

    @set_units(wavelengths = default_units.length)
    def at(self, wavelengths):
        spec = astropy_blackbody(self.temperature)(wavelengths) * self.solid_angle
        return make_spectrum_unit_field(wavelengths, spec)