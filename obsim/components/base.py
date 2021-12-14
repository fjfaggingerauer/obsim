from collections.abc import Sequence

__all__ = ['SimulationComponent',]

class SimulationComponent(object):
    def __init__(self, *args, **kwargs):
        pass

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

        raise NotImplementedError

    def forward(self, inp):
        '''
        Apply the component to the output from (a) previous component(s).
        '''

        raise NotImplementedError

    def evolve(self, evolve_parameters):
        '''
        Evolve the component between timesteps before propagating the next batch
        of inputs.

        Parameters
        ----------
        evolve_parameters : dict
            Dictionary containing parameters necessary for updating the
            component.
        '''
        pass

    def update_simulation(self, evolve_parameters):
        '''
        Pass information from the current component to the entire simulation by
        updating a simulation-wide dictionary of parameters.

        Parameters
        ----------
        evolve_parameters : dict
            Dictionary containing parameters to update the simulation.
        '''
        pass

    @property
    def number_of_inputs(self):
        return 1

    @property
    def previous_component(self):
        if hasattr(self, '_previous_component'):
            return self._previous_component
        else:
            return []

    @property
    def next_component(self):
        if hasattr(self, '_next_component'):
            return self._next_component
        else:
            return []

    @next_component.setter
    def next_component(self, val):
        self._next_component = val

    @previous_component.setter
    def previous_component(self, val):
        self._previous_component = val

    def apply(self):
        '''
        Function called by Simulation to propagate inputs from previous
        components. This function sets the output and gets the input from
        previous components.
        '''
        if len(self.previous_component) == 1:
            p = self.previous_component[0].output
            # previous component was None or source is outside fov
            if p is None:
                self.output = None
                return
        else:
            p = [s.output for s in self.previous_component]
            # previous component was None or source is outside fov
            if any([q is None for q in p]):
                self.output = None
                return

        self.output = self.forward(p)


    def __call__(self, previous_component):
        '''
        Link the component to a (set of) previous component(s). This initialises
        the current component to the properties of the previous component(s),
        and also links both the current component and the supplied component(s).

        Parameters
        ----------
        previous_component : SimulationComponent/sequence of SimulationComponent
            Single component or sequence of components. All must be derived
            from the SimulationComponent class.

        Returns
        -------
        self
            The current component, linked to the supplied component(s).
        '''
        if isinstance(previous_component, Sequence):
            for c in previous_component:
                if not isinstance(c, SimulationComponent):
                    raise ValueError("Previous component(s) must be derived from SimulationComponent.")

                self.previous_component += [c]
                c.next_component += [self]
                self.initialise_for(c)
        elif isinstance(previous_component, SimulationComponent):
            previous_component.next_component += [self]
            self.previous_component += [previous_component]
            self.initialise_for(previous_component)
        else:
            raise ValueError("Previous component must be derived from SimulationComponent or a sequence of objects derived from SimulationComponent.")

        if len(self.previous_component) > self.number_of_inputs:
            raise ValueError(f"Current number of inputs ({len(self.previous_component)}) exceeds the allowed number of inputs ({self.number_of_inputs}) for this component.")

        return self
