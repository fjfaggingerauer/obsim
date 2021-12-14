from ...components.base import SimulationComponent

__all__ = ['hcipyComponent']

class hcipyComponent(SimulationComponent):
    def __init__(self, *args, **kwargs):
        pass

    def forward_background(self, background):
        return background

    @property
    def input_grid(self):
        raise NotImplementedError

    @property
    def output_grid(self):
        raise NotImplementedError

    @property
    def input_grid_type(self):
        raise NotImplementedError

    @property
    def output_grid_type(self):
        raise NotImplementedError

    def apply(self):
        '''
        Function called by Simulation to propagate inputs from previous components.
        This function merely sets the output and gets the input from previous components.
        '''
        from .pupil_generator import hcipyBackground
        if len(self.previous_component) == 1:
            p = self.previous_component[0].output
            if p is None:
                self.output = None
                return
        else:
            p = [s.output for s in self.previous_component]
            if any([q is None for q in p]):
                self.output = None
                return
        
        if (isinstance(p, list) and all([isinstance(q, hcipyBackground) for q in p])) or isinstance(p, hcipyBackground):
            self.output = self.forward_background(p)
        else:
            self.output = self.forward(p)