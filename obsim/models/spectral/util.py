from ...util import set_units
from ...config import default_units, observing_bands
from ...field import Grid, SeparatedCoords, CartesianGrid, Field, UnstructuredCoords

import numpy as np
import astropy.constants as const
import warnings
import astropy.units as u

__all__  = ['make_spectrum_unit_field', 'make_wavelengths', 'convolve_to_resolution', 'doppler_shift_wavelengths']

@set_units(wavelengths=default_units.length)
def make_spectrum_unit_field(wavelengths, spectrum, convert_units_to_default=True):
    if convert_units_to_default:
        spectrum = spectrum.to(default_units.flux_wavelength_density, equivalencies = u.spectral_density(wavelengths))
    
    # check if wavelength grid is regular
    if np.any(np.diff(wavelengths) != np.diff(wavelengths)[0]):
        grid = Grid(UnstructuredCoords(wavelengths))
    else:
        warnings.warn("Regular unit grid not yet implemented, defaulting to unstructured unit coords...")
        grid = Grid(UnstructuredCoords(wavelengths))
    
    return Field(spectrum, grid)

@set_units(min_wavelength=default_units.length, max_wavelength=default_units.length)
def make_wavelengths(spectral_resolution=None, observing_band=None, min_wavelength=None,
                         max_wavelength=None, wavelength_number=None, oversampling=1, spacing='logarithmic', as_grid=False):
    '''
    Make a set of wavelengths in an observing band or between two wavelengths.
    The output can be given as an astropy.Quantity or a UnitGrid. Units of the 
    output will always default to default_units.length.

    A set of wavelengths can be generated either by providing a spectral resolution
    and observing band pair, or by providing a minimum wavelength, maximum wavelength
    and number of samples. If all are provided the observing band and wavelength number
    take precedence.

    Parameters
    ----------
    spectral_resolution : scalar, optional
        Spectral resolution of the output wavelength set.
    observing_band : str, optional
        Observing band of the output wavelength set, see config.observing_bands for
        the currently supported observing bands.
    min_wavelength : `astropy.Quantity`, optional
        Smallest wavelength in output. Must have unit equivalent to default_units.length.
    max_wavelength : `astropy.Quantity`, optional
        Largest wavelength in output. Must have unit equivalent to default_units.length.
    wavelength_number : int, optional
        Number of wavelengths in output.
    oversampling : scalar, optional
        Oversampling factor for the given spectral resolution. Is ignored if 
        wavelength_number is set. Default is 1.
    spacing : {'logarithmic', 'linear'}, optional
        Wavelength spacing in output. Default is 'logarithmic'.
    as_grid : bool, optional
        If True, returns a UnitGrid with the resulting wavelengths, otherwise
        returns an astropy.Quantity. Default is False.
    
    Returns
    -------
    output
        An `astropy.Quantity` or `UnitGrid` with the desired wavelength set.
        
    '''
    # TBD wavelength spacing for fixed spectral resolution

    # make sure we have a pair of min_wavelength, max_wavelength
    if max_wavelength is None:
        min_wavelength = None

    if spectral_resolution is None and wavelength_number is None:
        raise ValueError("Either spectral resolution or wavelength_number must be given")

    # get central_wavelength & bandwidth
    if observing_band is not None:
        try:
            central_wavelength, bandwidth = observing_bands[observing_band]
        except KeyError:
            raise ValueError(f"Band '{observing_band}' is not a supported observing band. See config.observing_bands for available observing bands.")

    elif min_wavelength is not None:
        central_wavelength = (max_wavelength + min_wavelength) / 2
        bandwidth = max_wavelength - min_wavelength

    else:
        raise ValueError("Either an observing band or a (min_wavelength, max_wavelength) pair must be provided.")

    # build grid
    if wavelength_number is None:
        N = int(spectral_resolution*oversampling * (bandwidth/central_wavelength).value + 0.5)
    else:
        N = wavelength_number

    cw = central_wavelength.to(default_units.length).value
    bw = bandwidth.to(default_units.length).value

    if spacing == 'logarithmic':
        wavelengths = np.logspace(np.log10(cw-bw/2), np.log10(cw+bw/2), N)
    elif spacing == 'linear':
        wavelengths = np.linspace(cw-bw/2, cw+bw/2, N)
    else:
        raise ValueError("Spacing '{spacing}' not supported.")
    
    if as_grid:
        return Grid(SeparatedCoords(wavelengths * default_units.length))
    else:
        return wavelengths * default_units.length

def convolve_to_resolution(spectrum, spectral_resolution, downsample=True):
    if spectrum.ndim != 1:
        raise ValueError(f"Supplied UnitField must be 1-dimensional, received a {spectrum.ndim}-dimensional input.")

    from scipy.ndimage import gaussian_filter1d

    eps = 1E-2
    
    R0 = spectrum.grid.coords[0][:-1]/np.diff(spectrum.grid.coords[0])

    # resample at largest resolution if variation through grid is too big
    if np.max(np.abs(R0)) > (1+eps)*np.min(np.abs(R0)):
        R0 = np.max(np.abs(R0))
        new_grid = make_wavelengths(min_wavelength = spectrum.grid.min(0), max_wavelength = spectrum.grid.max(0), spectral_resolution = R0)
        spectrum = spectrum.at(new_grid)
    
    # make kernel properties
    kernel_fwhm = np.mean(R0)/spectral_resolution
    kernel_std = (kernel_fwhm) / (2*np.sqrt(2*np.log(2)))

    # apply downsampling if required
    if downsample and int(kernel_fwhm/4) > 1:
        sampling = range(0, spectrum.size, int(kernel_fwhm/4))
    else:
        sampling = range(spectrum.size)
    
    # get filtered spectrum
    filtered_values = gaussian_filter1d(spectrum.value, kernel_std)[sampling]
    grid = CartesianGrid(SeparatedCoords(spectrum.grid.coords[0][sampling]))
    filtered_spectrum = Field(filtered_values * spectrum.unit, grid)

    return filtered_spectrum

@set_units(wavelengths=default_units.length, radial_velocity=default_units.velocity)
def doppler_shift_wavelengths(wavelengths, radial_velocity):
    beta = radial_velocity/const.c
    return wavelengths * np.sqrt((1+beta)/(1-beta))