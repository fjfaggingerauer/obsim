import hcipy as hp

from .base import hcipyComponent

__all__ = ['IdealCoronagraph']

class IdealCoronagraph(hcipyComponent):
    '''
    Ideal coronagraph
    '''
    def __init__(self, aperture_function, order=2):
        self.aperture = aperture_function
        self.order = order

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
        self.prev_component = prev_component
        self.grid = prev_component.output_grid
        self.coronagraph = hp.PerfectCoronagraph(self.aperture(self.grid), self.order)

    def forward(self, wf):
        return self.coronagraph.forward(wf)
