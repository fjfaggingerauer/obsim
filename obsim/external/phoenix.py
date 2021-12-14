import numpy as np
import os
import astropy.units as u
from astropy.io import fits
import os
import wget

from ..field import CartesianGrid, SeparatedCoords, Field

__all__ = ['get_PHOENIX_model']

def get_PHOENIX_model(temperature, log_g, fe_h=0, wl_lims=[0.5, 3]):
    if isinstance(temperature, u.Quantity):
        temperature = temperature.to(u.K).value
    sign_specifier = '+' if fe_h>0 else '-'
    t_val = int(200*np.round(temperature/200))
    log_g_val = 0.5*np.round(log_g/0.5)
    fe_h_val = 0.5*np.round(fe_h/0.5)
    fname = 'lte{0:05d}-{1:.2f}{2}{3:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(t_val, log_g_val, sign_specifier, np.abs(fe_h_val))

    fpath = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/'+fname
    data_path = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_path):
        print("Making local data folder...")
        os.path.mkdir(data_path)
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
    flux_density = hdu[0].data.astype(float) * u.erg/ (u.cm**3 * u.s)
    
    hdu = fits.open(wave_savepath)
    wavelengths = hdu[0].data.astype(float)*u.AA

    mask = (wavelengths>wl_lims[0]*u.micron)*(wavelengths<wl_lims[1]*u.micron)

    grid = CartesianGrid(SeparatedCoords(wavelengths[mask]))
    return Field(flux_density[mask], grid)