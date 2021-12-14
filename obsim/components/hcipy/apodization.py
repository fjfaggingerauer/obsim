import numpy as np
import hcipy as hp

from .base import hcipyComponent
from ...config import simulation_units
from ...field import Field

__all__ = ['Apodizer', 'WavefrontAberration', 'PowerLawAberration']

class Apodizer(hcipyComponent):
    def __init__(self, generator, grid_type):
        self.generator = generator
        self.grid_type = grid_type

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
    def filled_fraction(self):
        apod = self.apodizer.apodization
        return np.sum(np.abs(apod))/apod.grid.size
    
    def initialise_for(self, component):
        if component.output_grid_type != self.input_grid_type:
            raise ValueError(f"Component requires {self.input_grid_type} grid but previous component outputs a {component.output_grid_type} grid.")

        self.grid = component.output_grid
        self.apodizer = hp.Apodizer(self.generator(self.grid))
    
    def forward(self, inp):
        return self.apodizer.forward(inp)

    def forward_background(self, background):
        background.value = self.filled_fraction * background.value

        return background

class WavefrontAberration(hcipyComponent):
    '''
    Applies specified wavefront abberation to the incoming wavefront.
    wavefront_maps: UnitField or list of UnitFields with the wavefront errors to apply
    sequential: If True, the list of aberrations will be applied in order, if False a random
                entry will be selected at each timestep.
    '''
    def __init__(self, wavefront_maps=None, sequential=True):
        if not isinstance(wavefront_maps, list):
            if wavefront_maps.ndim==2:
                wavefront_maps= [wavefront_maps]
        self.wavefront_maps = wavefront_maps
        self.sequential = sequential
        self.iterator = 0

    def initialise_for(self, component):
        self.grid = component.output_grid

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

    def forward(self, wavefront):
        wavefront.electric_field *= np.exp(1j*2*np.pi*self.wf_abb/wavefront.wavelength)
        return wavefront

    def evolve(self, evolve_parameters):
        if len(self.wavefront_maps)>1:
            if self.sequential:
                self.iterator += 1
                if self.iterator==len(self.wavefront_maps):
                    self.iterator = 0
            else:
                self.iterator = np.random.sample(len(self.wavefront_maps))
            self.wf_abb = self.wavefront_maps[self.iterator].to(simulation_units.length).value


class PowerLawAberration(WavefrontAberration):
    def __init__(self, rms, power_law_index=-2.5, static=True):
        self.rms = rms
        self.power_law_index = power_law_index
        self.static = static

    def initialise_for(self, component):
        self.prev_component = component
        self.grid = component.grid
        self.aperture = hp.circular_aperture(component.grid.x.max())(self.grid)
        self.reset_abberation()

    def evolve(self, evolve_parameters):
        if not self.static:
            self.reset_abberation()

    def reset_abberation(self):
        abberation = hp.make_power_law_error(self.grid, ptv=1, diameter=self.grid.x.max(), 
                                            exponent=self.power_law_index)
        abberation = Field(self.rms * abberation/np.std(abberation[self.aperture==1]), abberation.grid)
        self.wf_abb = abberation.to(simulation_units.length).value