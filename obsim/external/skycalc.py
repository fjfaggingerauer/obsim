import numpy as np
import subprocess
import os
import astropy.units as u
from astropy.io import fits
import astropy.constants as const
import astropy.coordinates as c

from ..util import make_properties, calculate_airmass
from ..config import default_units
from ..field import Grid, CartesianGrid, Field, SeparatedCoords

__all__ = ['SkyCalcInterface']


class SkyCalcInterface(object):
    property_list = {
        'pwv_mode': {
            'type': str,
            'default': 'pwv',
            'allowed': ['pwv', 'season'],
        },
        'pwv': {
            'unit': default_units.length,
            'default': 3.5 * u.mm,
            'allowed': np.array([0.05, 0.01, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5,
                                 5.0, 7.5, 10.0, 20.0, 30.0]) * u.mm,
            'output_unit': u.mm,
        },
        'msolflux': {
            'unit': default_units.flux_frequency_density,
            'default': 130 * 0.01E6 * u.Jy,
            'limits': np.array([0, np.inf]) * u.Jy,
            'output_unit': 0.01E6 * u.Jy,
        },
        'incl_moon': {
            'type': bool,
            'default': True,
        },
        'moon_earth_dist': {
            'type': float,
            'default': 1.0,
            'limits': [0.91, 1.08],
        },
        'incl_starlight': {
            'type': bool,
            'default': True,
        },
        'incl_zodiacal': {
            'type': bool,
            'default': True,
        },
        # TBD: can this be calculated from observing_time + location?
        'ecl_lon': {
            'unit': u.deg,
            'default': 135 * u.deg,
            'limits': np.array([-180, 180]) * u.deg,
            'output_unit': u.deg,
        },
        # TBD: can this be calculated from observing_time + location?
        'ecl_lat': {
            'unit': u.deg,
            'default': 90 * u.deg,
            'limits': np.array([-90, 90]) * u.deg,
            'output_unit': u.deg,
        },
        'incl_loweratm': {
            'type': bool,
            'default': True,
        },
        'incl_upperatm': {
            'type': bool,
            'default': True,
        },
        'incl_airglow': {
            'type': bool,
            'default': True,
        },
        'incl_therm': {
            'type': bool,
            'default': False,
        },
        'therm_t1': {
            'unit': u.K,
            'default': 0.0 * u.K,
            'limits': np.array([0.0, np.inf])*u.K,
            'output_unit': u.K,
        },
        'therm_e1': {
            'type': float,
            'default': 0.0,
            'limits': [0, 1]
        },
        'therm_t2': {
            'unit': u.K,
            'default': 0.0 * u.K,
            'limits': np.array([0.0, np.inf])*u.K,
            'output_unit': u.K,
        },
        'therm_e2': {
            'type': float,
            'default': 0.0,
            'limits': [0, 1]
        },
        'therm_t3': {
            'unit': u.K,
            'default': 0.0 * u.K,
            'limits': np.array([0.0, np.inf])*u.K,
            'output_unit': u.K,
        },
        'therm_e3': {
            'type': float,
            'default': 0.0,
            'limits': [0, 1]
        },
        'vacair': {
            'type': str,
            'default': 'vac',
            'allowed': ['vac', 'air'],
        },
        'wmin': {
            'unit': default_units.length,
            'default': 300 * u.nm,
            'limits': np.array([3E2, 3E4])*u.nm,
            'output_unit': u.nm,
        },
        'wmax': {
            'unit': default_units.length,
            'default': 2E3 * u.nm,
            'limits': np.array([3E2, 3E4])*u.nm,
            'output_unit': u.nm,
        },
        'wgrid_mode': {
            'type': str,
            'default': 'fixed_wavelength_step',
            'allowed': ['fixed_spectral_resolution', 'fixed_wavelength_step']
        },
        'wdelta': {
            'unit': default_units.length,
            'default': 0.1 * u.nm,
            'limits': np.array([0, 3E4])*u.nm,
            'output_unit': u.nm,
        },
        'wres': {
            'type': float,
            'default': 20000.0,
            'limits': [0, 1e6],
        },
        'lsf_type': {
            'type': str,
            'default': 'none',
            'allowed': ['none', 'Gaussian', 'Boxcar']
        },
        'lsf_gauss_fwhm': {
            'type': float,
            'default': 5.0,
            'limits': [0, np.inf]
        },
        'lsf_boxcar_fwhm': {
            'type': float,
            'default': 5.0,
            'limits': [0, np.inf]
        },
        'observatory': {
            'type': str,
            'default': 'paranal',
            # 'default' : '2400',
            'allowed': ['paranal', 'lasilla', '3060m'],
            # 'allowed' : ['2400', '2640', '3060'],
        },
        'target_alt': {
            'unit': default_units.angle,
            'default': 45*u.deg,
            'limits': np.array([-90, 90])*u.deg,
            'output_unit': u.deg,
        },
        'moon_alt': {
            'unit': default_units.angle,
            'default': 45*u.deg,
            'limits': np.array([-90, 90])*u.deg,
            'output_unit': u.deg,
        },
        'moon_target_sep': {
            'unit': default_units.angle,
            'default': 45*u.deg,
            'limits': np.array([0, 180])*u.deg,
            'output_unit': u.deg,
        },
        'moon_sun_sep': {
            'unit': default_units.angle,
            'default': 90*u.deg,
            'limits': np.array([0, 360])*u.deg,
            'output_unit': u.deg,
        },
        'moon_earth_dist': {
            'type': float,
            'default': 1.0,
            'limits': [0.91, 1.08],
        },
    }

    def __init__(self, telescope=None, data_path='./data/', **kwargs):
        self.telescope = telescope
        self.data_path = data_path
        make_properties(self, self.property_list, kwargs)

        self.parameter_filename = 'skycalc_input.txt'
        self.output_filename = 'skycalc_output.fits'
        self.grid = None
        self.moon_position = None
        self.sun_position = None

    @property
    def observation_time(self):
        if self.telescope is None:
            return
        return self.telescope.observation_time

    @property
    def pointing(self):
        if self.telescope is None:
            return
        return self.telescope.physical_pointing

    @property
    def location(self):
        if self.telescope is None:
            return
        return self.telescope.location

    @property
    def sun_position(self):
        if self.telescope is None:
            return
        if self._sun_position is None:
            geocentric_sun_pos = c.get_sun(self.observation_time)
            self._sun_position = geocentric_sun_pos.transform_to(
                c.AltAz(obstime=self.observation_time,
                        location=self.location))

        return self._sun_position

    @sun_position.setter
    def sun_position(self, val):
        if val is not None:
            raise ValueError("Sun position is determined automatically and can\
 only be reset to None.")
        self._sun_position = None

    @property
    def moon_position(self):
        if self.telescope is None:
            return
        if self._moon_position is None:
            geocentric_moon_pos = c.get_moon(self.observation_time)
            self._moon_position = geocentric_moon_pos.transform_to(c.AltAz(
                obstime=self.observation_time, location=self.location))

        return self._moon_position

    @moon_position.setter
    def moon_position(self, val):
        if val is not None:
            raise ValueError("Moon position is determined automatically and \
can only be reset to None.")
        self._moon_position = None

    def get_airmass(self):
        if self.telescope is None:
            return calculate_airmass(90*u.deg - self.target_alt)
        return calculate_airmass(90*u.deg - self.pointing.alt)

    def get_moon_sun_sep(self):
        if self.telescope is None:
            return self.moon_sun_sep.to_value(u.deg)
        return self.moon_position.separation(self.sun_position).to_value(u.deg)

    def get_moon_target_sep(self):
        if self.telescope is None:
            return self.moon_target_sep.to_value(u.deg)
        return self.moon_position.separation(self.pointing).to_value(u.deg)

    def get_moon_alt(self):
        if self.telescope is None:
            return self.moon_alt.to_value(u.deg)
        return self.moon_position.alt.to_value(u.deg)

    def get_moon_earth_dist(self):
        raise NotImplementedError()

    def _call_skycalc_interface(self, verbose=False):
        call_list = ['skycalc_cli', '-i', self.data_path
                     + self.parameter_filename, '-o', self.data_path
                     + self.output_filename]

        if verbose:
            call_list += ['--verbose']
        try:
            subprocess.run(call_list)
        except FileNotFoundError:
            raise RuntimeError("SkyCalc executable not found, either \
skycalc_cli is not installed or it is not on path.")

    def _get_output(self):
        if not os.path.exists(self.data_path + self.output_filename):
            raise RuntimeError("Can't find the created SkyCalc output file, \
either the output path is wrong or SkyCalc did not produce an output file.")

        hdu = fits.open(self.data_path + self.output_filename)
        output_unit = 1./u.s/u.m**2/u.micron/u.arcsec**2 * const.h * const.c

        if self.telescope is None:
            output_unit *= 1*u.arcsec**2
        else:
            output_unit *= self.telescope.field_of_view**2

        wl = hdu[1].data['LAM']*u.nm

        flux = hdu[1].data['FLUX'] * (output_unit/wl).to(
            default_units.flux_wavelength_density)
        transmission = hdu[1].data['TRANS']

        if self.grid is None:
            self.grid = CartesianGrid(SeparatedCoords(wl))

        sky_emission = Field(flux, self.grid)
        sky_transmission = Field(transmission, self.grid)

        self.grid = None
        self.moon_position = None
        self.sun_position = None

        return sky_emission, sky_transmission

    def _get_property(self, param):
        param_dict = self.property_list[param]
        val = getattr(self, param)

        # check whether current value is allowed
        if 'allowed' in param_dict.keys():
            if val not in param_dict['allowed']:
                raise ValueError(f"Parameter '{param}' is set to '{val}', but \
must be one of {param_dict['allowed']}.")
        elif 'limits' in param_dict.keys():
            if val < param_dict['limits'][0] or val > param_dict['limits'][1]:
                raise ValueError(f"Value for parameter '{param}' is '{val}', \
which is outside the allowed range of '{param_dict['limits']}'.")

        return val, param_dict

    def _make_val_str(self, val, param_dict):
        # convert value to form required for input file
        if isinstance(val, str):
            val_str = val
        elif isinstance(val, bool):
            if val:
                val_str = "Y"
            else:
                val_str = "N"
        elif isinstance(val, float):
            val_str = str(val)
        elif isinstance(val, u.Quantity):
            raw_value = (val / param_dict['output_unit']).to(
                u.dimensionless_unscaled)
            val_str = str(raw_value.value)
        else:
            val_str = ""

        return val_str

    def _check_separation_constraint(self):
        if self.telescope is None:
            target_alt = self.target_alt
            moon_alt = self.moon_alt
        else:
            target_alt = self.pointing.alt
            moon_alt = self.moon_position.alt

        airmass = self.get_airmass()
        z = (90*u.deg - target_alt).to_value(u.deg)
        z_moon = (90*u.deg - moon_alt).to_value(u.deg)
        rho = self.get_moon_target_sep()

        if np.abs(z-z_moon) > rho or np.abs(z+z_moon) < rho:
            raise RuntimeError(f"The physical constraint |z-z_moon| <= rho \
<= |z+z_moon| is not satisfied.")
        if airmass < 1.0 or airmass > 3.0:
            raise RuntimeError(f"SkyCalc requires an airmass in the range \
[1.0, 3.0], but current pointing has an airmass of {airmass:.3g}.")

    def _build_input_file(self):
        output_list = []

        if os.path.exists(self.data_path + self.output_filename):
            os.remove(self.data_path + self.output_filename)

        # get all necessary parameters from the property list
        for param in self.property_list.keys():
            if param not in ['airmass', 'moon_sun_sep', 'moon_target_sep',
                             'moon_alt', 'target_alt']:
                val, param_dict = self._get_property(param)
                val_str = self._make_val_str(val, param_dict)

                output_list.append(f"{param} : {val_str}")

        self._check_separation_constraint()

        # add other parameters from functions
        output_list.append(f"airmass : {self.get_airmass()}")
        output_list.append(f"moon_sun_sep : {self.get_moon_sun_sep()}")
        output_list.append(f"moon_target_sep : {self.get_moon_target_sep()}")
        output_list.append(f"moon_alt : {self.get_moon_alt()}")

        os.makedirs(self.data_path, exist_ok=True)
        full_path = self.data_path + self.parameter_filename

        with open(full_path, "w") as parameter_file:
            for p in output_list:
                parameter_file.write(p + '\n')

    def set_wavelengths(self, wavelengths, mode=None):
        '''
        Set minimum/maximum wavelength and wavelength step based on supplied
        set of wavelengths.
        '''
        if isinstance(wavelengths, Grid):
            wavelengths = wavelengths.coords[0]

        if mode is not None:
            self.wgrid_mode = mode

        self.wmin = wavelengths.min()
        self.wmax = wavelengths.max()
        dlam = np.diff(wavelengths)

        if self.wgrid_mode == 'fixed_wavelength_step':
            self.wdelta = dlam.min()
        elif self.wgrid_mode == 'fixed_spectral_resolution':
            self.wres = (wavelengths[:-1]/dlam).to_value(
                u.dimensionless_unscaled).max()

    def make_fields(self, wavelengths=None, mode=None):
        if wavelengths is not None:
            self.set_wavelengths(wavelengths, mode)
        self._build_input_file()
        self._call_skycalc_interface()
        return self._get_output()
