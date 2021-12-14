__all__ = ['plot']

from ..config import default_units
#from ..sources import Spectrum
from ..field import Field

import matplotlib.pyplot as plt
import astropy.units as u

'''
def plot_spectrum(spectrum, wavelength_unit=default_units.length, spectrum_unit=default_units.flux_wavelength_density, **kwargs):
    if not isinstance(spectrum, Spectrum):
        raise ValueError("Supplied spectrum must be instance of 'Spectrum'")

    wl = spectrum.wavelengths.to(wavelength_unit).value
    flx = spectrum.to(spectrum_unit).value

    plt.plot(wl, flx, **kwargs)
'''

def plot(field, grid_units=None, field_unit=None, xlabel="", ylabel="", **kwargs):
    if not isinstance(field, Field):
        raise TypeError("Input field must be a UnitField.")
    
    if field.grid.ndim > 1: # create multiple lines over other axes eventually
        raise NotImplementedError()
    
    if grid_units is None:
        grid_units = field.grid.coords.units[0]
    if field_unit is None:
        field_unit = field.unit
    
    x = field.grid.coords[0].to_value(grid_units)
    y = field.to_value(field_unit)

    if grid_units != u.dimensionless_unscaled:
        xlabel = xlabel + f"({grid_units})"
    if field_unit != u.dimensionless_unscaled:
        ylabel = ylabel + f"({field_unit})"

    plt.plot(x,y, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
