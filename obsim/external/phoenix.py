import numpy as np
import os
import astropy.units as u
from astropy.io import fits
import os
import wget

from ..field import CartesianGrid, SeparatedCoords, Field
from ..util import strip_units

# __all__ = ['get_PHOENIX_model']
__all__ = ['get_PHOENIX_spectrum']


def get_PHOENIX_model(temperature, log_g, fe_h=0, wl_lims=[0.5, 3]):
    if isinstance(temperature, u.Quantity):
        temperature = temperature.to(u.K).value
    sign_specifier = '+' if fe_h > 0 else '-'
    t_val = int(200*np.round(temperature/200))
    log_g_val = 0.5*np.round(log_g/0.5)
    fe_h_val = 0.5*np.round(fe_h/0.5)
    fname = 'lte{0:05d}-{1:.2f}{2}{3:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(t_val, log_g_val, sign_specifier, np.abs(fe_h_val))

    fpath = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/'+fname
    data_path = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_path):
        print("Making local data folder...")
        os.mkdir(data_path)
    savepath = os.path.join(data_path, fname)
    wave_path = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS//WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
    wave_savepath = os.path.join(data_path, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
    if not os.path.exists(wave_savepath):
        print('Downloading wavelength grid from '+wave_path)
        wget.download(wave_path, wave_savepath)
    if not os.path.exists(savepath):
        print('Downloading PHOENIX spectrum from '+fpath)
        wget.download(fpath, savepath)

    hdu = fits.open(savepath)
    flux_density = hdu[0].data.astype(float) * u.erg / (u.cm**3 * u.s)
    
    hdu = fits.open(wave_savepath)
    wavelengths = hdu[0].data.astype(float)*u.AA

    mask = (wavelengths > wl_lims[0]*u.micron)*(wavelengths < wl_lims[1]*u.micron)

    grid = CartesianGrid(SeparatedCoords(wavelengths[mask]))
    return Field(flux_density[mask], grid)


@strip_units(T=u.K, min_wl=u.AA, max_wl=u.AA)
def get_PHOENIX_spectrum(T: u.Quantity = 6000 * u.K, log_g: float = 4.0,
                         fe_h: float = 0, min_wl: u.Quantity = 0.5 * u.micron,
                         max_wl: u.Quantity = 3 * u.micron,
                         verbose: bool = False,
                         save_location: str = 'data'):
    '''
    Get a stellar spectrum from the PHOENIX library of spectra.
    The spectrum is downloaded through the ftp portal of the database
    using the `wget` package.

    The given parameters will be rounded to the nearest available PHOENIX
    model. This means `T` will be rounded to a multiple of 200, and `log_g`
    and `fe_h` to the nearest half.

    Parameters
    ----------
    T : astropy.units.Quantity
        Temperature of the star in Kelvin.
    log_g : float
        Logarithm of the stellar surface gravity.
    fe_h : float
        Logarithm of the stellar metallicity.
    min_wl : astropy.units.Quantity
        Minimum wavelength of the output Field in micron.
    max_wl : astropy.units.Quantity
        Maximum wavelength of the output Field in micron.
    verbose : bool
        Print progress statements during the function. Default is False.
    save_location : str
        Path relative to script file where the data is stored. This function
        adds two fits files to this location. Default is 'data'

    Returns
    -------
    spectrum : Field
        The resulting stellar spectrum as a Field. The resulting field is
        1d, with the only axis being the wavelength axis in Angstrom. The data
        has units of erg/cm^3/s.
    '''
    # check wavelength limits
    if min_wl < 500.0:
        raise ValueError(f"Minimum wavelength {min_wl:.3g} A is not supported,\
 must be at least 500 A.")
    if max_wl > 54999.75:
        raise ValueError(f"Maximum wavelength {max_wl:.3g} A is not supported,\
 must be at most 54999.75 A.")

    # toggle wget progress bar display
    if verbose:
        wget_bar = wget.bar_adaptive
        print("Starting to get PHOENIX spectrum...")
    else:
        wget_bar = None

    # data location urls
    ftp_path_head = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/'
    spectrum_url = 'PHOENIX-ACES-AGSS-COND-2011/Z-0.0/'
    wavelength_fname = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

    # ensure save path for fits exists
    save_folder = os.path.join(os.getcwd(), save_location)
    if not os.path.exists(save_folder):
        if verbose:
            print(f"'{save_folder}' does not exist, creating it...")
        os.mkdir(save_folder)

    # round input parameters
    sign_str = '+' if fe_h > 0 else '-'
    t_val = int(200 * np.round(T / 200))
    log_g_val = 0.5 * np.round(log_g / 0.5)
    fe_h_val = 0.5 * np.round(fe_h / 0.5)

    spectrum_fname = f'lte{t_val:05d}-{log_g_val:.2f}{sign_str}{fe_h_val:.1f}\
.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

    if verbose:
        print(f"Closest available model has T = {t_val:05d} K, log_g = \
{log_g_val:.2f}, and fe_h = {fe_h_val:.1f}.")

    # get wavelength data
    wavelength_save_path = os.path.join(save_folder, wavelength_fname)
    if not os.path.exists(wavelength_save_path):
        if verbose:
            print("No existing wavelength fits found, downloading one...")
        wavelength_full_url = ftp_path_head + wavelength_fname
        wget.download(wavelength_full_url, wavelength_save_path, wget_bar)

    wavelength_hdu = fits.open(wavelength_save_path)
    wavelength = wavelength_hdu[0].data.astype(float)
    wavelength_hdu.close()

    # get spectrum data
    spectrum_save_path = os.path.join(save_folder, spectrum_fname)
    if not os.path.exists(spectrum_save_path):
        if verbose:
            print("No existing spectrum found with given parameters, \
downloading one...")
        spectrum_full_url = ftp_path_head + spectrum_url + spectrum_fname
        wget.download(spectrum_full_url, spectrum_save_path, wget_bar)

    spectrum_hdu = fits.open(spectrum_save_path)
    spectrum = spectrum_hdu[0].data.astype(float) * u.erg / (u.cm**3 * u.s)
    spectrum_hdu.close()

    # create output Field
    mask = (wavelength >= min_wl)*(wavelength <= max_wl)
    grid = CartesianGrid(SeparatedCoords(wavelength[mask] * u.AA))
    output = Field(spectrum[mask], grid)

    if verbose:
        print("Done creating output, returning...\n")

    return output
