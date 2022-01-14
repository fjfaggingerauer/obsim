import numpy as np
import astropy.units as u
import wget
import ssl
import os

from ..field import CartesianGrid, SeparatedCoords, Field
from ..util import strip_units

__all__ = ['get_BT_SETTL_spectrum', 'get_BT_SETTL_model']

# TODO: spectral model of BT_SETTL

def get_BT_SETTL_model(temperature, log_g, wl_lims=[0.5, 3]):
    ssl._create_default_https_context = ssl._create_unverified_context
    #The file names contain the main parameters of the models:
    #lte{Teff/10}-{Logg}{[M/H]}a[alpha/H].GRIDNAME.7.spec.gz/bz2/xz
    t_val = int(np.round(temperature/100))
    log_g_val = 0.5*np.round(log_g/0.5)
    if t_val<12:
        fname = 'lte{0:03d}-{1:.1f}-0.0a+0.0.BT-Settl.spec.7.bz2'.format(t_val, log_g_val)
    else:
        fname = 'lte{0:03d}.0-{1:.1f}-0.0a+0.0.BT-Settl.spec.7.xz'.format(t_val, log_g_val)

    data_path = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_path):
        print("Making local data folder...")
        os.path.mkdir(data_path)
    fpath = os.path.join(data_path, fname)
    decompressed_fpath = fpath[:-4]
    if not os.path.exists(decompressed_fpath):
        if t_val<12:
            url = 'https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011/SPECTRA/'+fname
        else:
            url = 'https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/SPECTRA/'+fname
        print('Downloading BT-SETTL spectra from:', url)
        file = wget.download(url,fpath)
    
        if t_val<12:
            with open(fpath, 'rb') as file:
                import bz2
                data = file.read()
                decompressed_data = bz2.decompress(data)
        else:
            import lzma
            decompressed_data = lzma.open(fpath).read()
        with open(decompressed_fpath,'wb') as file:
            file.write(decompressed_data)
    
    with open(decompressed_fpath, 'r') as file:
        data = file.readlines()
    wl = np.zeros(len(data))
    flux = np.zeros(len(data))

    for i,line in enumerate(data):
        split_line = line.split()
        approx_wl = float(split_line[0][:6])
        if (approx_wl>wl_lims[0]*1e4)*(approx_wl<wl_lims[1]*1e4):
            try:
                wl[i] = float(split_line[0])*1e-4
                flux[i] = 10**(float(split_line[1].replace('D','E'))-8)
            except:
                double_splitted = split_line[0].split('-')
                wl[i] = float(double_splitted[0])*1e-4
                if len(split_line)==2:
                    flux[i] = 10**(float(double_splitted[1].replace('D','E'))-8)
                else:
                    flux[i] = 10**(float((double_splitted[1]+'-'+double_splitted[2]).replace('D','E'))-8)
    if wl_lims is not None:
        mask = (wl>wl_lims[0])*(wl<wl_lims[1])
        wl, flux = wl[mask], flux[mask]
    sorting = np.argsort(wl)
    wl = wl[sorting]
    flux = flux[sorting]
    
    #return Spectrum(flux*u.erg/u.s/u.cm**2/u.AA, wl*u.micron)
    grid = CartesianGrid(SeparatedCoords(wl*u.micron))
    return Field(flux * u.erg/u.s/u.cm**2/u.AA, grid)


@strip_units(T=u.K, min_wl=u.micron, max_wl=u.micron)
def get_BT_SETTL_spectrum(T: u.Quantity = 6000 * u.K, log_g: float = 4.0,
                          min_wl: u.Quantity = 0.5 * u.micron,
                          max_wl: u.Quantity = 3.0 * u.micron,
                          verbose: bool = False, save_location: str = 'data'):
    '''
    Get a spectrum from the BT_SETTL library of spectra. The spectrum is
    downloaded using the `wget` package.

    The given parameters will be rounded to the nearest available BT_SETTL
    model. This means `T` will be rounded to a multiple of 100, and `log_g`
    to the nearest half.

    Parameters
    ----------
    T : astropy.units.Quantity
        Temperature of the star in Kelvin.
    log_g : float
        Logarithm of the surface gravity.
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
        has units of erg/s/cm^2/Angstrom.
    '''
    def decompress_bz2(p):
        import bz2

        with open(p, 'rb') as file:
            raw_data = file.read()
            data = bz2.decompress(raw_data)

        return data

    def decompress_lzma(p):
        import lzma

        file = lzma.open(p)
        data = file.read()

        return data

    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    except ImportError:
        pass

    if min_wl is None:
        min_wl = 0
    if max_wl is None:
        max_wl = np.inf

    # round input parameters
    t_val = int(np.round(T/100))
    log_g_val = 0.5 * np.round(log_g/0.5)

    # ensure save path for fits exists
    save_folder = os.path.join(os.getcwd(), save_location)
    if not os.path.exists(save_folder):
        if verbose:
            print(f"'{save_folder}' does not exist, creating it...")
        os.mkdir(save_folder)

    # make urls
    if t_val < 12:
        url = 'https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011/SPECTRA/'
        fname = f'lte{t_val:03d}-{log_g_val:.1f}-0.0a+0.0.BT-Settl.spec.7.bz2'
        decompress = decompress_bz2
    else:
        url = 'https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/\
SPECTRA/'
        fname = f'lte{t_val:03d}.0-{log_g_val:.1f}-0.0a+0.0.BT-Settl.spec.7.xz'
        decompress = decompress_lzma

    file_path = os.path.join(save_folder, fname)
    unzipped_file_path = file_path[:-4]

    # ensure data has been downloaded
    if not os.path.exists(unzipped_file_path):
        wget.download(url + fname, file_path)
        file_data = decompress(file_path)

        with open(unzipped_file_path, 'wb') as f:
            f.write(file_data)

    with open(unzipped_file_path, 'rb') as f:
        file_data = f.readlines()

    # fill out data
    wavelengths = np.zeros(len(file_data))
    fluxes = np.zeros(len(file_data))

    for ii, line in enumerate(file_data):
        split_line = line.decode().split()
        approx_wl = float(split_line[0][:6])
        if (approx_wl > min_wl*1E4) and (approx_wl < max_wl*1E4):
            try:
                wavelengths[ii] = float(split_line[0]) * 1E-4
                fluxes[ii] = 10**(float(split_line[1].replace('D', 'E')) - 8)
            except ValueError:
                double_splitted = split_line[0].split('-')
                wavelengths[ii] = float(double_splitted[0]) * 1E-4
                if len(split_line) == 2:
                    fluxes[ii] = 10**(float(double_splitted[1]
                                            .replace('D', 'E')) - 8)
                else:
                    fluxes[ii] = 10**(float((double_splitted[1] + '-'
                        + double_splitted[2]).replace('D', 'E')) - 8)

    # polish data to output format
    mask = (wavelengths > min_wl) * (wavelengths < max_wl)
    wavelengths, fluxes = wavelengths[mask], fluxes[mask]

    sorting = np.argsort(wavelengths)
    wavelengths = wavelengths[sorting] * u.micron
    fluxes = fluxes[sorting] * u.erg/u.s/u.cm**2/u.AA

    grid = CartesianGrid(SeparatedCoords(wavelengths))

    return Field(fluxes, grid)
