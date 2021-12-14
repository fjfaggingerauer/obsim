from .base import SimulationComponent

__all__ = ['ComponentSequence']

class ComponentSequence(SimulationComponent):
    def __init__(self, components):
        self.components = components
    
    def __getitem__(self, index):
        return self.components[index]
    
    def __getattr__(self, name):
        return getattr(self.components[-1], name)
    
    def initialise_for(self, component):
        '''
        Initialise component based on the properties of the supplied component.
        This includes setting grids or building optical components.

        Parameters
        ----------
        component : SimulationComponent
            Component in front of the current component, which already has been
            initialised.
        '''
        # link self.components[0] to self without setting self.next_component
        self.components[0].initialise_for(component)
        self.components[0].previous_component = [self]

        x = self.components[0]
        for c in self.components[1:]:
            x = c(x)

    def forward(self, val):
        '''
        Apply the component to the output from (a) previous component(s).
        '''

        self.output = val
        for c in self.components:
            c.apply()
        
        if hasattr(self.components[-1], 'output'):
            #self.output = self.components[-1].output
            return self.components[-1].output
    
    def evolve(self, evolve_parameters):
        '''
        Evolve the component between timesteps before propagating the next batch of inputs.

        Parameters
        ----------
        evolve_parameters : dict
            Dictionary containing parameters necessary for updating the component.
        '''
        for c in self.components:
            c.evolve(evolve_parameters)

    def update_simulation(self, evolve_parameters):
        '''
        Pass information from the current component to the entire simulation by updating
        a simulation-wide dictionary of parameters.
        '''
        for c in self.components:
            c.update_simulation(evolve_parameters)