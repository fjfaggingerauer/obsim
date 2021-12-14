from .base import SpectralModel
from .analytical import FlatSpectrum
from .util import make_spectrum_unit_field
from ...config import default_units
from ...util import make_properties, set_units

import numpy as np

__all__ = ['AbsorptionLineSpectrum', 'EmissionLineSpectrum']

class AbsorptionLineSpectrum(SpectralModel):
    property_list = {
        'central_wavelengths' : {
            'unit' : default_units.length,
        },
        'line_widths' : {
            'unit' : default_units.length,
        },
        'line_strengths' : {
            'unit' : 1/default_units.length,
            'default' : None,
        },
        'line_profile' : {
            'type' : str,
            'default' : 'gaussian',
        },
        'background' : {
            'type' : SpectralModel,
            'default' : None
        },
        'path_length' : {
            'unit' : default_units.length,
            'default' : 1
        },
    }
    def __init__(self, central_wavelengths, line_widths, **kwargs):
        make_properties(self, self.property_list, kwargs)
        self.central_wavelengths = central_wavelengths
        self.line_widths = line_widths
        self.args = None
    
    @property
    def function(self):
        def profile_lorentz(wl0, FWHM, wls):
            x = ((wls-wl0)/(FWHM/2)).value
            return 1/(1+x**2)

        def profile_gaussian(wl0, FWHM, wls):
            x = ((wls-wl0)/(FWHM/2)).value
            return np.exp(-np.log(2)*x**2)
        
        if self.line_profile == 'gaussian':
            profile = profile_gaussian
        elif self.line_profile == 'lorentzian':
            profile = profile_lorentz
        else:
            raise NotImplementedError(f"Line profile '{self.line_profile}' not supported.")

        if self.line_strengths is not None:
            line_strengths = self.line_strengths
        else:
            line_strengths = np.ones(len(self.central_wavelengths)) / default_units.length
        
        def f(wavelengths):
            ab = np.zeros(wavelengths.shape) / default_units.length
            for wl0, dwl, H in zip(self.central_wavelengths, self.line_widths, line_strengths):
                ab += H * profile(wl0, dwl, wavelengths)
            
            return np.exp(-self.path_length * ab)
        
        return f
    
    def at(self, wavelengths):
        if self.background is None:
            bg = FlatSpectrum()
        else:
            bg = self.background
        
        return bg.at(wavelengths) * self.function(wavelengths)

class EmissionLineSpectrum(SpectralModel):
    property_list = {
        'central_wavelengths' : {
            'unit' : default_units.length,
        },
        'line_widths' : {
            'unit' : default_units.length,
        },
        'line_strengths' : {
            'unit' : default_units.flux_wavelength_density,
            'default' : None,
        },
        'line_profile' : {
            'type' : str,
            'default' : 'gaussian',
        },
    }
    def __init__(self, central_wavelengths, line_widths, **kwargs):
        make_properties(self, self.property_list, kwargs)
        self.central_wavelengths = central_wavelengths
        self.line_widths = line_widths
        self.args = None
    
    @property
    def function(self):
        def profile_lorentz(wl0, FWHM, wls):
            x = ((wls-wl0)/(FWHM/2)).value
            return 1/(1+x**2)

        def profile_gaussian(wl0, FWHM, wls):
            x = ((wls-wl0)/(FWHM/2)).value
            return np.exp(-np.log(2)*x**2)
        
        if self.line_profile == 'gaussian':
            profile = profile_gaussian
        elif self.line_profile == 'lorentzian':
            profile = profile_lorentz
        else:
            raise NotImplementedError(f"Line profile '{self.line_profile}' not supported.")

        if self.line_strengths is not None:
            line_strengths = self.line_strengths
        else:
            line_strengths = np.ones(len(self.central_wavelengths)) * default_units.flux_wavelength_density
        
        def f(wavelengths):
            ab = np.zeros(wavelengths.shape) * default_units.flux_wavelength_density
            for wl0, dwl, H in zip(self.central_wavelengths, self.line_widths, line_strengths):
                ab += H * profile(wl0, dwl, wavelengths)
            
            return ab
        
        return f
    
    @set_units(wavelengths=default_units.length)
    def at(self, wavelengths):
        spec = self.function(wavelengths)
        return make_spectrum_unit_field(wavelengths, spec)
