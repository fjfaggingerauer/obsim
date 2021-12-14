from ...util import set_units, strip_units
from ...config import default_units, simulation_units
from .base import hcipyComponent

import numpy as np
import hcipy as hp
import astropy.constants as const
import copy
import astropy.units as u
from collections.abc import Sequence

__all__ = ['Detector3D', 'Detector', 'PhysicalDetector']

_ch = (const.c * const.h).to(simulation_units.energy * simulation_units.length).value

class Detector(hcipyComponent):
    def __init__(self):
        self.accumulated_charge = 0
        self.charge_rate = 0
    
    @property
    def input_grid(self):
        return None
    
    @property
    def input_grid_type(self):
        return None
    
    @property
    def output_grid(self):
        return None
    
    @property
    def output_grid_type(self):
        return None
    
    def initialise_for(self, component):
        pass
    
    def forward(self, val):
        #val = self.previous_component[0].output
        if isinstance(val, Sequence):
            for v in val:
                self.integrate(v)
        else:
            self.integrate(val)
    
    def forward_background(self, background):
        self.integrate_background(background)

    def integrate_background(self, background):
        photon_energy = _ch / background.wavelength
        photon_rate = background.value / photon_energy
        self.charge_rate += photon_rate
    
    def integrate(self, val):
        #photon_energy = (const.c*const.h/(val.wavelength*simulation_units.length)).to(simulation_units.energy).value
        photon_energy = _ch / val.wavelength
        photon_rate = val.power / photon_energy # in simulation_units.frequency

        self.charge_rate += photon_rate

    def evolve(self, evolve_parameters):
        dt = evolve_parameters['timestep']
        if isinstance(dt, u.Quantity):
            dt = dt.to(simulation_units.time).value
        
        self.accumulated_charge += self.charge_rate * dt
        self.charge_rate = 0
    
    def read_out(self):
        if isinstance(self.accumulated_charge, int):
            return self.accumulated_charge
        output = self.accumulated_charge.copy()
        self.accumulated_charge = 0

        return output
    

class Detector3D(Detector):
    @strip_units(wavelengths=default_units.length)
    def __init__(self, detector, wavelengths):
        self.wavelengths = wavelengths
        self.wavelength_diffs = np.diff(wavelengths,prepend=wavelengths[0]-(wavelengths[1]-wavelengths[0]))
        self.detector = detector
    
    @property
    def input_properties(self):
        return None
    
    @property
    def input_properties(self):
        return None
    
    def initialise_for(self, component):
        self.detector_dict = {'{0:.10E}'.format(wavelength): copy.deepcopy(self.detector) for wavelength in self.wavelengths}
        for key in self.detector_dict.keys():
            self.detector_dict[key].initialise_for(component)

    def integrate(self, val):
        self.detector_dict['{0:.10E}'.format(val.wavelength)].integrate(val)
    
    def integrate_background(self, val):
        self.detector_dict['{0:.10E}'.format(val.wavelength)].integrate_background(val)
    
    def evolve(self, evolve_parameters):
        return [det.evolve(evolve_parameters) for det in self.detector_dict.values()]
    
    def read_out(self, destructive=True):
        output = np.array([det.read_out(destructive).shaped for det in self.detector_dict.values()])
        return output
    
class PhysicalDetector(Detector):
    #@strip_units(dark_current_rate = simulation_units.frequency)
    @set_units(dark_current_rate = default_units.frequency)
    def __init__(self, read_noise=0, dark_current_rate=0, include_photon_noise=True, well_depth=np.inf, gain=1, activation_function=None, quantum_efficiency=None):
        self.read_noise = read_noise
        #self.dark_current_rate = dark_current_rate.to(simulation_units.frequency).value
        self.dark_current_rate = dark_current_rate
        self.include_photon_noise = include_photon_noise
        self.well_depth = well_depth
        self.gain = gain
        self.activation_function = activation_function
        self.quantum_efficiency = quantum_efficiency

        if callable(gain): #TBD detector grid automatization
            raise NotImplementedError

    def initialise_for(self, component):
        self.grid = component.output_grid
        self.accumulated_charge = self.grid.zeros()
        self.charge_rate = self.grid.zeros()*simulation_units.frequency

    @property
    def activation_function(self):
        return self._activation_function
    
    @activation_function.setter
    def activation_function(self, f):
        def linear_activation(e_num, well_depth):
            return np.minimum(e_num, well_depth)

        if f is None:
            self._activation_function = linear_activation
        else:
            if not callable(f):
                raise ValueError("Activation function must be callable.")
            self._activation_function = f
    
    @property
    def quantum_efficiency(self):
        return self._quantum_efficiency
    
    @quantum_efficiency.setter
    def quantum_efficiency(self, f):
        if isinstance(f, float) or isinstance(f, int):
            self._quantum_efficiency = lambda x: f
        elif f is None:
            self._quantum_efficiency = lambda x: 1.0
        else:
            if not callable(f):
                raise ValueError("Activation function must be callable.")
            self._quantum_efficiency = f
    
    def integrate(self, wavefront, weight=1):
        if not hasattr(wavefront, 'power') or not hasattr(wavefront, 'wavelength'):
            raise ValueError("Input must have attributes 'power' and 'wavelength' when integrated with this detector type.")
        
        photon_energy = _ch/(wavefront.wavelength)
        photon_rate = wavefront.power / photon_energy * simulation_units.frequency
        QE = self.quantum_efficiency(wavefront.wavelength * simulation_units.length)

        self.charge_rate += photon_rate * weight * QE

    def integrate_background(self, background, weight=1):
        if not hasattr(background, 'wavelength'):
            raise ValueError("Input must have attribute 'wavelength' when integrated with this detector type.")
        
        photon_energy = _ch/(background.wavelength)
        photon_rate = background.value / photon_energy * simulation_units.frequency
        QE = self.quantum_efficiency(background.wavelength * simulation_units.length)

        self.charge_rate += photon_rate * weight * QE

    def evolve(self, evolve_parameters):
        dt = evolve_parameters['timestep']
        #if isinstance(dt, u.Quantity):
        #    dt = dt.to(simulation_units.time).value
        self.accumulated_charge += (self.charge_rate * dt).value
        self.accumulated_charge += (self.dark_current_rate * dt).value
        self.charge_rate = self.grid.zeros()*simulation_units.frequency
    
    # adapted from hcipy's NoisyDetector.readout
    def read_out(self, destructive=True):
        output_field = self.accumulated_charge.copy()

        # Reset detector
        if destructive:
            self.accumulated_charge = self.grid.zeros()
            self.charge_rate = self.grid.zeros()*simulation_units.frequency

        # Adding photon noise.
        if self.include_photon_noise:
            output_field = hp.large_poisson(output_field, thresh=1e6)

        # Adding flat field errors. 
        #output_field *= self.flat_field

        # Adding read-out noise.
        output_field += np.random.normal(loc=0, scale=self.read_noise, size=output_field.size)

        # Add pixel activation function
        output_field = self.activation_function(output_field, self.well_depth)

        # Add gain (TBD gain variation as flat-field noise)
        output_field *= self.gain

        return output_field