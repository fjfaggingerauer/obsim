from ..models import FlatSpectrum, SpectralModel
from ..util import set_units, make_properties
from ..config import default_units

import numpy as np
import astropy.units as u
import astropy.coordinates as c

__all__ = ['Source', 'Background']

# https://docs.astropy.org/en/stable/coordinates/index.html
# https://docs.astropy.org/en/stable/generated/examples/coordinates/plot_obs-planning.html#sphx-glr-generated-examples-coordinates-plot-obs-planning-py

class Source(object):
    @set_units(wavelengths=default_units.length)
    def __init__(self, wavelengths=None, location=None, physical_location=None, spectral_model=None, polarization_model=None, extent_model=None):
        self.physical_location = physical_location
        self.wavelengths = wavelengths
        self.location = location
        self.spectral_model = spectral_model
        self.polarization_model = polarization_model
        self.extent_model = extent_model

        #self._spectral_model = None
        #self._polarization_model = None
        #self._extent_model = None
        

    @property
    def location(self):
        if hasattr(self, '_location') and self._location is not None:
            return self._location
        if self._physical_location is not None:
            return self._physical_location
        
        return [0,0]*default_units.angle
        

    @property
    def wavenumbers(self):
        return 2*np.pi/self.wavelengths
    
    @property
    def physical_location(self):
        return self._physical_location
    
    @physical_location.setter
    def physical_location(self, val=None):
        if not isinstance(val, c.SkyCoord) and val is not None:
            raise ValueError("Physical location must be set with a SkyCoord")
        self._physical_location = val

    #'''
    @location.setter
    def location(self, value=None):
        if value is None:
            self._location = value
        elif isinstance(value, u.Quantity):
            self._location = value.to(default_units.angle)
        elif isinstance(value, c.SkyCoord):
            self._physical_location = value
        else:
            try:
                self._location = value * default_units.angle
            except Exception as e:
                raise ValueError(f"Failed to set location, original error was: {str(e)}")
    #'''
    '''
    @location.setter
    @set_units(value = default_units.angle)
    def location(self, value=None):
        self._location = value
    #'''
    @property
    def spectral_model(self):
        if self._spectral_model is None:
            q = FlatSpectrum()
            q.initialise_for(self)
            return q
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, spectral_model):
        self._spectral_model = spectral_model

        if self._spectral_model is not None:
            if isinstance(self._spectral_model, SpectralModel):
                self._spectral_model.initialise_for(self)
            else:
                raise ValueError("Must set 'spectral_model' with a SpectralModel or derived class.")

    @property
    def spectrum(self):
        if self.wavelengths is None:
            raise Exception("Must set wavelengths before accessing this attribute")
        return self.spectral_model(self.wavelengths)

    @property
    def polarization_model(self):
        return self._polarization_model

    @polarization_model.setter
    def polarization_model(self, polarization_model):
        self._polarization_model = polarization_model
        if self._polarization_model is not None:
            if isinstance(self._polarization_model, None): # TBD implement PolarizationModel + derived classes
                self._polarization_model.initialise_for(self)
            else:
                raise ValueError("Must set 'polarization_model' with a PolarizationModel or derived class.")

    @property
    def polarization(self):
        if self.wavelengths is None:
            raise Exception("Must set wavelengths before accessing this attribute")
        self.polarization_model(self.wavelengths)

    @property
    def extent(self):
        return None # TBD extended sources?

    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    @set_units(wavelengths=default_units.length)
    def wavelengths(self, wavelengths=None):
        self._wavelengths = wavelengths

    def evolve(self, evolve_parameters):
        pass

    def reset(self):
        pass

class Background(Source):
    property_list = {
        'temperature' : {
            'unit' : default_units.temperature,
            'default' : 293 * u.K
        },
        'scale' : {
            'unit' : default_units.angular_area,
            'default' : 1
        },
    }
    def __init__(self, wavelengths=None, **kwargs):
        make_properties(self, self.property_list, kwargs)
        super().__init__(wavelengths, None)
    
    @property
    def location(self):
        raise AttributeError("The location of a ThermalBackground cannot be accessed.")
    
    @location.setter
    def location(self, v):
        pass