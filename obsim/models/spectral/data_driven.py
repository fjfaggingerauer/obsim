import numpy as np
import astropy.units as u

from ...util import set_units
from ...config import default_units
from ...field import Field
from ...external import get_PHOENIX_model, get_BT_SETTL_model
from .base import SpectralModel
from .util import make_spectrum_unit_field

__all__ = ['InterpolatedSpectrum', 'FunctionSpectrum', 'BT_SETTL_Model']

class FunctionSpectrum(SpectralModel):
    def __init__(self, function, args=None):
        self.function = function
        self.args = args

    @set_units(wavelengths=default_units.length)
    def at(self, wavelengths):
        if self.args is None:
            spec = self.function(wavelengths)
        else:
            spec = self.function(wavelengths, *self.args)
        
        return make_spectrum_unit_field(wavelengths, spec)

class InterpolatedSpectrum(SpectralModel):
    @set_units(wavelength=default_units.length, flux_density=[default_units.flux_wavelength_density, default_units.flux_frequency_density])
    def __init__(self, spectrum, wavelengths=None, mode='linear'):
        if isinstance(spectrum, Field):
            self.wavelengths = spectrum.grid[0]
            self.flux_density = spectrum
        else:
            self.wavelengths = wavelengths
            self.flux_density = make_spectrum_unit_field(wavelengths, spectrum)
        
        self.mode = mode

    def interpolate(self, new_wavelengths):
        if self.mode == 'linear':
            new_spec = np.interp(new_wavelengths, self.wavelengths, self.flux_density)
            new_spec = make_spectrum_unit_field(new_wavelengths, new_spec)
        elif self.mode == 'flux_conserving':
            raise NotImplementedError
        else:
            raise NotImplementedError

        return new_spec

    def at(self, wavelengths):
        from ...sources import Background
        if isinstance(self.source, Background):
            received_flux_density = self.flux_density
        else:
            received_flux_density = self.flux_density * (self.source.radius**2/(4*self.source.distance**2))
        return self.interpolate(wavelengths)

class PHOENIX_Model(InterpolatedSpectrum):
    def __init__(self, spectral_resolution=None):
        self.spectral_resolution = spectral_resolution

    def initialise_for(self, source):
        log_g = np.log10(source.surface_gravity.to(u.cm/u.s**2).value)
        wl, spec = get_PHOENIX_model(source.temperature, log_g, spectral_resolution=self.spectral_resolution,
                    wl_lims = [0.8*np.min(source.wavelengths), 1.2*np.max(source.wavelengths)])

        spec = spec * source.radius**2/(4*source.distance**2)
        super().__init__(wl, spec)
        super().initialise_for(source)
        
class BT_SETTL_Model(InterpolatedSpectrum):
    def __init__(self, spectral_resolution=None):
        self.spectral_resolution = spectral_resolution

    def initialise_for(self, source):
        log_g = np.log10(source.surface_gravity.to(u.cm/u.s**2).value)
        wl, spec = get_BT_SETTL_model(source.temperature, log_g, spectral_resolution=self.spectral_resolution,
                    wl_lims = [0.8*np.min(source.wavelengths), 1.2*np.max(source.wavelengths)])
        spec = spec * source.radius**2/(4*source.distance**2)
        super().__init__(wl, spec)
        super().initialise_for(source)