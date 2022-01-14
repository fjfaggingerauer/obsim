import pathlib
import sys

p = pathlib.Path(__file__).absolute().parents[1]
sys.path.append(str(p))

import obsim as obs
import astropy.units as u


def get_model_through_func():
    f = obs.get_PHOENIX_spectrum(6000*u.K, 4.44,
                                 min_wl=0.06*u.micron, max_wl=3*u.micron)

    assert(u.isclose(f.grid.coords[0].min(), 600.0*u.AA))
    assert(u.isclose(f.grid.coords[0].max(), 29999.875*u.AA))
    assert(u.isclose(f.max(), 1379438134558720.0 * u.erg / (u.cm**3 * u.s)))
    assert(u.isclose(f.min(), 1.7785752959298406e-07 * u.erg / (u.cm**3 * u.s)))


def get_model_through_spectralmodel():
    wavelengths = obs.make_wavelengths(1000, 'R')
    s = obs.Star(wavelengths)
    s.temperature = 8000*u.K

    spec = obs.PhoenixSpectrum()
    s.spectral_model = spec

    f = spec.flux_density

    assert(u.isclose(f.grid.coords[0].min(), 5890.01*u.AA))
    assert(u.isclose(f.grid.coords[0].max(), 7270.0*u.AA))
    assert(u.isclose(f.min(), 7.597299634012905e-05 * u.J/(u.m**3 * u.s)))
    assert(u.isclose(f.max(), 0.0003631993313922163 * u.J/(u.m**3 * u.s)))


if __name__ == '__main__':
    get_model_through_func()
    get_model_through_spectralmodel()
