import numpy as np
import hcipy as hp
import astropy.constants as const
import astropy.units as u
from astropy.io import fits

from ...field import Field, CartesianGrid, SeparatedCoords
from ...util import set_units, make_properties
from ...config import simulation_units, default_units
from ...models import convolve_to_resolution
from .base import hcipyComponent
from .propagation import EffectiveThroughput

__all__ = ['Atmosphere', 'AtmosphericTransmission', 'get_sky_emission']

class AtmosphericTransmission(EffectiveThroughput):
    '''
    Applies atmospheric transmission.
    Data is taken from ESO Skycalc (http://www.eso.org/observing/etc/bin/simu/skycalc)
    With airmass = 1.5, PWV = 2.5 mm
    '''
    def __init__(self,  wavelengths, spectral_resolution=None,path='../data/skytable.fits', airmass=1.5):
        # estimate spectral resolution from wavelength sampling if not given
        if spectral_resolution is None:
            spectral_resolution = np.median(wavelengths[:-1]/np.diff(wavelengths))
        
        hdu = fits.open(path) #From ESO Skycalc
        wl = hdu[1].data['lam']*u.nm
        T = hdu[1].data['trans']
        
        transmission = Field(T, CartesianGrid(SeparatedCoords(wl)))
        transmission = convolve_to_resolution(transmission, spectral_resolution)

        self.airmass = airmass
        super().__init__(transmission.at(CartesianGrid(SeparatedCoords(wavelengths))))
    
    def _get_throughput(self, wavelength):
        #return (self.airmass/1.5) * self.throughput.interpolated(wavelength*simulation_units.length).value
        return (self.airmass/1.5) * self.throughput.at(wavelength*simulation_units.length).value
    
    '''
    def forward(self, wf_in):
        wf_out = wf_in.copy()
        wf_out.electric_field *= np.sqrt((self.airmass/1.5)*self.throughput.interpolated(wf_in.wavelength*simulation_units.length).value)
        return wf_out

    def forward_background(self, bg_in):
        T = (self.airmass/1.5)*self.throughput.interpolated(bg_in.wavelength*simulation_units.length).value 
        bg_in.value = T * bg_in.value

        return bg_in
    '''
    def evolve(self, evolve_params):
        if 'airmass' in evolve_params.keys():
            self.airmass = evolve_params['airmass']

def get_sky_emission(fov, spectral_resolution=None, path='../data/skytable.fits', airmass=1.5):
    #TODO: Add varying airmass data
    '''
    Data is taken from ESO Skycalc (http://www.eso.org/observing/etc/bin/simu/skycalc)
    With airmass = 1.5, PWV = 2.5 mm
    '''
    hdu = fits.open(path) #From ESO Skycalc
    wl = hdu[1].data['lam']*u.nm
    flux = hdu[1].data['flux'] * 1./u.s/u.m**2/u.micron/u.arcsec**2 *const.h*const.c/wl * fov**2
    #sky_emission = Spectrum(flux, wl)
    sky_emission = Field(flux, CartesianGrid(SeparatedCoords(wl)))
    if spectral_resolution is not None:
        sky_emission = convolve_to_resolution(sky_emission, spectral_resolution)

    print(type(sky_emission))
    print(sky_emission.grid)
    return sky_emission



def fried_parameter_to_seeing(fried_parameter, reference_wavelength):
    return 0.98 * reference_wavelength / fried_parameter * u.rad # from hp.fried_parameter_to_seeing
def seeing_to_fried_parameter(seeing, reference_wavelength):
    return 0.98 * reference_wavelength / seeing.to(u.rad).value # from hp.seeing_to_fried_parameter

# tau_0 = 0.314 * r_0/v_eff
def get_coherence_time(fried_parameter, v_eff):
    return 0.314 * fried_parameter/v_eff
def get_v_eff(coherence_time, fried_parameter):
    return 0.314 * fried_parameter / coherence_time
def get_fried_parameter(coherence_time, v_eff):
    return v_eff * coherence_time / 0.314

def get_cn_squared(fried_parameter, reference_wavelength):
    return hp.Cn_squared_from_fried_parameter(fried_parameter.to(u.m).value, reference_wavelength.to(u.m).value)

class Atmosphere(hcipyComponent):
    property_list = {
        'seeing' : {
            'unit' : default_units.angle,
            'functions' : [(fried_parameter_to_seeing, ('fried_parameter', 'reference_wavelength'))]
        },
        'fried_parameter' : {
            'unit' : default_units.length,
            'functions' : [(seeing_to_fried_parameter, ('seeing', 'reference_wavelength')),
                           (get_fried_parameter, ('coherence_time', 'effective_wind_velocity'))],
            'default' : 0.2 * u.m
        },
        'reference_wavelength' : {
            'unit' : default_units.length,
            'default' : 500 * u.nm
        },
        'coherence_time' : {
            'unit' : default_units.time,
            'default' : 5E-3*u.s,
            'functions' : [(get_coherence_time, ('fried_parameter', 'effective_wind_velocity'))]
        }, # tau_0
        'layer_number' : {
            'type' : int,
            'default' : 1
        },
        'effective_wind_velocity' : {
            'unit' : default_units.velocity,
            'functions' : [(get_v_eff, ('coherence_time', 'fried_parameter'))],
        },
        'outer_scale' : {
            'unit' : default_units.length,
            'default' : 40 * u.m
        },
        'total_Cn_squared' : {
            'functions' : [(get_cn_squared, ('fried_parameter', 'reference_wavelength'))]
        },
        'wind_directions' : {},
    }
    def __init__(self, **kwargs):
        make_properties(self, self.property_list, kwargs)
        self.t = 0
    
    @property
    def input_grid(self):
        return self.grid
    
    @property
    def input_grid_type(self):
        return 'pupil'
    
    @property
    def output_grid(self):
        return self.grid
    
    @property
    def output_grid_type(self):
        return 'pupil'

    def initialise_for(self, prev_component):
        if prev_component.output_grid_type != 'pupil':
            raise ValueError(f"Atmosphere must receive a 'pupil' input grid, which is incompatible with a '{prev_component.output_grid_type}' grid.")
        
        self.grid = prev_component.output_grid
        
        # distribute Cn2 randomly between layers
        rand_floats = np.random.uniform(0, 1, size = self.layer_number)
        weights = rand_floats/np.sum(rand_floats)
        layer_Cn2s = weights * self.total_Cn_squared
        
        # get wind direction per layer
        if self.wind_directions is None:
            layer_angles = np.random.uniform(0, 2*np.pi, size = self.layer_number)
        else:
            wind_directions = np.array(self.wind_directions)
            if wind_directions.ndim == 0:
                layer_angles = np.tile(wind_directions, self.layer_number)
            elif wind_directions.ndim == 1:
                if len(wind_directions) != self.layer_number:
                    raise ValueError("Different number of wind directions than number of layers.")
                layer_angles = wind_directions
            else:
                raise ValueError("Wind directions must be a number or a 1d sequence")

        # distribute wind speeds randomly between layers
        rand_floats =  np.random.uniform(0, 1, size = self.layer_number)
        weights = ((rand_floats/np.sum(rand_floats)) * (self.total_Cn_squared/layer_Cn2s))**(3./5)
        layer_velocities = weights * self.effective_wind_velocity.to(simulation_units.velocity).value

        # build multi-layer atmosphere
        wind_velocities = [v for v, theta in zip(layer_velocities, layer_angles)]
        #wind_velocities = [v * np.array([np.cos(theta), np.sin(theta)]) for v, theta in zip(layer_velocities, layer_angles)]
        atmospheric_layers = [hp.InfiniteAtmosphericLayer(self.grid, Cn2, v, 2, use_interpolation = True) for Cn2, v in zip(layer_Cn2s, wind_velocities)]
        self.atmosphere = hp.MultiLayerAtmosphere(atmospheric_layers)

    def forward(self, wf):
        return self.atmosphere.forward(wf)
    
    def update_simulation(self, evolve_parameters):
        evolve_parameters['atmosphere_opd'] = self.atmosphere.phase_for(1)/(2*np.pi)

    def evolve(self, evolve_parameters):
        dt = evolve_parameters['timestep']
        if isinstance(dt, u.Quantity):
            dt = dt.to(simulation_units.time).value
        self.t += dt
        self.atmosphere.evolve_until(self.t)
        evolve_parameters['atmosphere_opd'] = self.atmosphere.phase_for(1)/(2*np.pi)
