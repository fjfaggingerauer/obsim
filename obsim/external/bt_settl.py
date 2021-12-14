import numpy as np
import astropy.units as u
import wget
import ssl
import os

from ..field import CartesianGrid, SeparatedCoords, Field

__all__ = ['get_BT_SETTL_model']

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