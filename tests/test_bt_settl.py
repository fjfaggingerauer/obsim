import pathlib
import sys

p = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(p))

import obsim as obs
import astropy.units as u

import matplotlib.pyplot as plt


def get_model_through_func():
    f = obs.get_BT_SETTL_spectrum(6000*u.K, 4.44,
                                  min_wl=0.06*u.micron, max_wl=3*u.micron)

    assert(u.isclose(f.grid.coords[0].min(), 0.0600049 * u.micron))
    assert(u.isclose(f.grid.coords[0].max(), 2.99999 * u.micron))
    assert(u.isclose(f.max(), 14190575.216890898 * u.erg/(u.cm**2*u.s*u.AA)))
    assert(u.isclose(f.min(), 1.15664477416565e-18 * u.erg/(u.cm**2*u.s*u.AA)))


def get_model_through_spectralmodel():
    wavelengths = obs.make_wavelengths(1000, 'R')
    s = obs.Star(wavelengths)
    s.temperature = 6000*u.K

    spec = obs.BT_SETTLSpectrum()
    s.spectral_model = spec

    f = spec.flux_density.to(obs.default_units.flux_wavelength_density)

    assert(u.isclose(f.grid.coords[0].max(), 0.727*u.micron))
    assert(u.isclose(f.grid.coords[0].min(), 0.58901*u.micron))
    assert(u.isclose(f.max(), 0.00012766825575257653 * u.J/(u.m**3*u.s)))
    assert(u.isclose(f.min(), 1.4999712152709616e-05 * u.J/(u.m**3*u.s)))

if __name__ == '__main__':
    get_model_through_func()
    get_model_through_spectralmodel()
