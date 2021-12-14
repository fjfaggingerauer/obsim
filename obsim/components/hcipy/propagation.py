import hcipy as hp
import numpy as np
import astropy.units as u

from .base import hcipyComponent
from ...util import set_units
from ...config import default_units, simulation_units
from ...field import Field

__all__ = ['FraunhoferPropagator', 'EffectiveThroughput', 'make_ideal_passband_filter']

class FraunhoferPropagator(hcipyComponent):
    @set_units(f = default_units.length)
    def __init__(self, output_grid=None, f=None, propagator=None):
        self._output_grid = output_grid
        self.f = f
        self.propagator = propagator

        self._output_grid_type = None
        self._input_grid = None
        self._input_grid_type = None
    
    @property
    def input_grid(self):
        return self._input_grid
    
    @property
    def input_grid_type(self):
        return self._input_grid_type
    
    @property
    def output_grid(self):
        return self._output_grid
    
    @property
    def output_grid_type(self):
        return self._output_grid_type

    def initialise_for(self, component):
        if self.propagator is None:
            self._input_grid = component.output_grid

            if component.output_grid_type == 'pupil':
                pgrid = component.output_grid
                fgrid = self.output_grid
                self._output_grid_type = 'focal'
                self._input_grid_type = 'pupil'
            elif component.output_grid_type == 'focal':
                pgrid = self.output_grid
                fgrid = component.output_grid
                self._output_grid_type = 'pupil'
                self._input_grid_type = 'focal'
            else:
                raise ValueError(f"FraunhoferPropagator received a '{component.output_properties['grid_type']}' grid as input, but can only accept 'pupil' or 'focal' grids.")

            self.propagator = hp.FraunhoferPropagator(pgrid, fgrid, self.f.to(simulation_units.length).value)
        else:
            pgrid = self.propagator.get_input_grid(None,None)
            fgrid = self.propagator.get_output_grid(None,None)

            if component.output_grid_type == 'pupil':
                if hash(component.output_grid) != hash(pgrid):
                    raise ValueError("Output grid from previous component is inconsistent with the input grid of this propagator.")
                self._output_grid = fgrid
                self._output_grid_type = 'focal'
                self._input_grid = pgrid
                self._input_grid_type = 'pupil'
            elif component.output_grid_type == 'focal':
                if hash(component.output_grid) != hash(fgrid):
                    raise ValueError("Output grid from previous component is inconsistent with the input grid of this propagator.")
                self._output_grid = pgrid
                self._output_grid_type = 'pupil'
                self._input_grid = fgrid
                self._input_grid_type = 'focal'
            else:
                raise ValueError(f"FraunhoferPropagator received a '{component.output_grid_type}' grid as input, but can only accept 'pupil' or 'focal' grids.")
    
    def forward(self, inp):
        if self.output_grid_type == 'focal':
            return self.propagator.forward(inp)
        elif self.output_grid_type == 'pupil':
            return self.propagator.backward(inp)
        else:
            raise RuntimeError

class EffectiveThroughput(hcipyComponent):
    '''
    This component applies an effective wavelength-dependent throughput, while
    leaving the propagated fields otherwise unaltered.

    Parameters
    ----------
    throughput : float, Field, function
        Throughput per wavelength. This can either be a float for a constant
        throughput,  a Field of throughput per wavelength, or a function that 
        takes the wavelength in simulation_units.length as its argument.
    '''
    @set_units(wavelengths=default_units.length)
    def __init__(self, throughput, wavelengths=None):
        def make_constant_throughput(val):
            def f(wavelength):
                return val
            return f
        
        self.function = None
        self.throughput = None

        if callable(throughput):
            self.function = throughput
        elif isinstance(throughput, float):
            self.function = make_constant_throughput(throughput)
        #elif isinstance(throughput, Spectrum):
        #    self.throughput = throughput
        elif isinstance(throughput, Field):
            if throughput.unit != u.dimensionless_unscaled:
                raise ValueError(f"Throughput must be unitless, input has unit '{throughput.unit}'.")
            if throughput.grid.ndim != 1:
                raise ValueError(f"Throughput must be a 1-dimensional UnitField, input is {throughput.grid.ndim}-dimensional.")
            self.throughput = throughput
        else:
            raise ValueError(f"Throughput of type {type(throughput)} is not supported.")
    
    @property
    def input_grid(self):
        return self.grid
    
    @property
    def input_grid_type(self):
        return self.grid_type
    
    @property
    def output_grid(self):
        return self.grid
    
    @property
    def output_grid_type(self):
        return self.grid_type
    
    def _get_throughput(self, wavelength):
        if self.function is not None:
            return self.function(wavelength)
        
        #if isinstance(self.throughput, Spectrum):
        #    return self.throughput.interpolated(wavelength * simulation_units.length).value
        elif isinstance(self.throughput, Field):
            return self.throughput.at(wavelength * simulation_units.length).value
        else:
            raise ValueError("Unknown type of throughput.")

    def initialise_for(self, prev_component):
        self.prev_component = prev_component
        self.grid = prev_component.output_grid
        self.grid_type = prev_component.output_grid_type
    
    def forward(self, wf_in):
        wf_out = wf_in.copy()
        T = self._get_throughput(wf_in.wavelength)
        wf_out.electric_field *= np.sqrt(T)
        return wf_out
    
    def forward_background(self, bg_in):
        T = self._get_throughput(bg_in.wavelength)
        bg_in.value = T * bg_in.value

        return bg_in

@set_units(min_wavelength=default_units.length, max_wavelength=default_units.length)
def make_ideal_passband_filter(min_wavelength, max_wavelength):
    min_wl = min_wavelength.to_value(simulation_units.length)
    max_wl = max_wavelength.to_value(simulation_units.length)
    
    def f(wavelength):
        if wavelength < min_wl or wavelength > max_wl:
            return 0.0
        return 1.0
    
    return EffectiveThroughput(f)