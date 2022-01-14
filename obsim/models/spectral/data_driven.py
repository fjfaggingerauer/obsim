import numpy as np
import astropy.units as u
from astropy import constants as const

from ...util import set_units
from ...config import default_units
from ...field import Field
from ...external import get_PHOENIX_spectrum, get_BT_SETTL_model
from .base import SpectralModel
from .util import make_spectrum_unit_field

__all__ = ['InterpolatedSpectrum', 'FunctionSpectrum', 'bt_settlSpectrum',
           'PhoenixSpectrum']


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
            self.wavelengths = spectrum.grid.coords[0]
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


class PhoenixSpectrum(InterpolatedSpectrum):
    def __init__(self, data_path: str = 'data',
                 verbose_initialization: bool = False):

        self.data_path = data_path
        self.verbose = verbose_initialization

    def initialise_for(self, source):
        # set doppler shifted wavelengths
        if hasattr(source, 'radial_velocity'):
            beta = source.radial_velocity/const.c
            doppler_factor = np.sqrt((1-beta)/(1+beta))
        else:
            doppler_factor = 1

        kwargs = {'min_wl': doppler_factor * source.wavelengths.min(),
                  'max_wl': doppler_factor * source.wavelengths.max(),
                  'save_location': self.data_path,
                  'verbose': self.verbose}

        # find other parameters
        if hasattr(source, 'temperature'):
            kwargs['T'] = source.temperature
        if hasattr(source, 'surface_gravity'):
            sg = source.surface_gravity.to(u.cm/u.s**2)
            kwargs['log_g'] = np.log10(sg.value)
        if hasattr(source, 'metallicity'):
            kwargs['fe_h'] = source.metallicity

        # get distance-corrected spectrum
        spectrum = get_PHOENIX_spectrum(**kwargs)

        if hasattr(source, 'radius') and hasattr(source, 'distance'):
            spectrum = spectrum * source.radius**2/(4*source.distance**2)

        super().__init__(spectrum)
        super().initialise_for(source)


class bt_settlSpectrum(InterpolatedSpectrum):
    def __init__(self, spectral_resolution=None):
        self.spectral_resolution = spectral_resolution

    def initialise_for(self, source):
        log_g = np.log10(source.surface_gravity.to(u.cm/u.s**2).value)
        wl, spec = get_BT_SETTL_model(source.temperature, log_g, spectral_resolution=self.spectral_resolution,
                    wl_lims=[0.8*np.min(source.wavelengths), 1.2*np.max(source.wavelengths)])
        spec = spec * source.radius**2/(4*source.distance**2)
        super().__init__(wl, spec)
        super().initialise_for(source)