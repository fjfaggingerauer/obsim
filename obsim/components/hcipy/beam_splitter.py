import numpy as np
import hcipy as hp
from collections.abc import Sequence

from .base import hcipyComponent
from ..base import SimulationComponent
from ...config import simulation_units
from ...field import Field
from .propagation import EffectiveThroughput

__all__ = ['BeamSplitter', 'BeamCombiner']

class BeamSplitterOutput(SimulationComponent):
    def __init__(self, parent):
        self.parent = parent
        self.output = None
    
    def __getattr__(self, item):
        return getattr(self.parent, item)
    
    @property
    def next_component(self):
        return self.parent.next_component
    
    @next_component.setter
    def next_component(self, val): 
        self.parent.next_component = val


class BeamSplitter(hcipyComponent):
    '''
    Beam splitter that distributes its light to two arms based on
    the supplied throughput. Use the `.arm1` and `.arm2` attributes
    as components for the output of this component, not the BeamSplitter
    instance itself.

    Parameters
    ----------
    throughput : function, float, or Field
        Throughput to the first arm, similar to an `EffectiveThroughput`
        instance. If a function must take a wavelength in 
        simulation_units.length as its input.
    '''
    def __init__(self, throughput):
        self.arm1 = BeamSplitterOutput(self)
        self.arm2 = BeamSplitterOutput(self)

        EffectiveThroughput.__init__(self, throughput)

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
        elif isinstance(self.throughput, Field):
            return self.throughput.at(wavelength * simulation_units.length).value
        else:
            raise ValueError("Unknown type of throughput.")

    def initialise_for(self, prev_component):
        self.prev_component = prev_component
        self.grid = prev_component.output_grid
        self.grid_type = prev_component.output_grid_type
    
    def forward(self, wf_in):
        wf_out1 = wf_in.copy()
        wf_out2 = wf_in.copy()
        T = self._get_throughput(wf_in.wavelength)
        wf_out1.electric_field *= np.sqrt(T)
        wf_out2.electric_field *= np.sqrt(1-T)
        return wf_out1, wf_out2
    
    def forward_background(self, bg_in):
        T = self._get_throughput(bg_in.wavelength)
        bg2 = bg_in.copy()
        bg_in.value = T * bg_in.value
        bg2.value = (1-T) * bg2.value

        return bg_in, bg2
    
    @property
    def output(self):
        raise AttributeError("""Access the 'arm1' and 'arm2' attributes for the 
        output of this component.""")

    def apply(self):
        '''
        Function called by Simulation to propagate inputs from previous components.
        This function merely sets the output and gets the input from previous components.
        '''
        from .pupil_generator import hcipyBackground
        if len(self.previous_component) == 1:
            p = self.previous_component[0].output
            if p is None:
                self.arm1.output = self.arm2.output = None
                return
        else:
            p = [s.output for s in self.previous_component]
            if any([q is None for q in p]):
                self.arm1.output = self.arm2.output = None
                return
        
        if (isinstance(p, list) and all([isinstance(q, hcipyBackground) for q in p])) or isinstance(p, hcipyBackground):
            self.arm1.output, self.arm2.output = self.forward_background(p)
        else:
            self.arm1.output, self.arm2.output = self.forward(p)


class BeamCombiner(hcipyComponent):
    def __init__(self, path_differences=None, path_difference_evolve=None):
        self.path_differences = path_differences
        self.evolve_function = path_difference_evolve
    
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
    
    @property
    def number_of_inputs(self):
        return int(1E8)

    def initialise_for(self, prev_component):
        self.grid = prev_component.output_grid
        self.grid_type = prev_component.output_grid_type
    
    def evolve(self, evolve_parameters):
        if self.evolve_function is not None:
            self.path_differences = self.evolve_function(evolve_parameters['timestep'])

    def forward(self, wavefronts):
        if not isinstance(wavefronts, Sequence):
            wavefronts = [wavefronts]
        res = wavefronts[0].copy()
        res.electric_field *= 0
        for ii, wf in enumerate(wavefronts):
            if self.path_differences is not None:
                k = 2*np.pi/wf.wavelength
                c = np.exp(1j*k*self.path_differences[ii])
            else:
                c = 1
            res.electric_field += wf.electric_field * c

        return res

    def forward_background(self, backgrounds):
        if not isinstance(backgrounds, Sequence):
            backgrounds = [backgrounds]
        output = backgrounds[0].copy()
        output.value = sum([bg.value for bg in backgrounds])

        return output 
