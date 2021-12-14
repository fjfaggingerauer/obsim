from ...util import set_units
from ...config import default_units
from ..base import *

import astropy.constants as const
import numpy as np

__all__ = ['SpectralModel', 'SpectralModelSum']

class SpectralModelSum(ModelSum):
    def __add__(self, m):
        if not isinstance(m, SpectralModel) and not isinstance(m, SpectralModelSum):
            raise ValueError(f"Addition not supported between a Model and {type(m)}.")
        
        if isinstance(m, SpectralModelSum):
            self.models = self.models + m.models
        else:
            self.models = self.models + [m]
        
        return self

class SpectralModel(Model):
    @set_units(wavelength=default_units.length)
    def __call__(self, wavelengths):
        if (hasattr(self, 'source') and hasattr(self.source, 'radial_velocity')) and self.source.radial_velocity is not None:
            beta = self.source.radial_velocity/const.c
            doppler_factor = np.sqrt((1-beta)/(1+beta))
        else:
            doppler_factor = 1
        
        return self.at(doppler_factor * wavelengths)
    
    def __add__(self, m):
        if not isinstance(m, SpectralModel) and not isinstance(m, SpectralModelSum):
            raise ValueError(f"Addition not supported between a Model and {type(m)}.")

        if isinstance(m, SpectralModelSum):
            return m + self
        else:
            return SpectralModelSum([self, m])