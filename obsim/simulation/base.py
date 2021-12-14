__all__ = ['Simulation']

from ..util import set_units, strip_units
from ..config import default_units, simulation_units
from ..components import SimulationComponent

from collections.abc import Sequence
import numpy as np
from tqdm import tqdm
import time
from pathos import pools as mp

class Simulation(object):
    def __init__(self, first_component=None, final_component=None, source_list=None):
        self.source_list = source_list
        #self.final_component = final_component
        self.first_component = first_component

    @property
    def first_component(self):
        return self._first_component
  
    def make_component_slices(self, first_component):
        def get_next_components(component):
            if component is None:
                return []
            if isinstance(component, SimulationComponent):
                if len(component.next_component) == 0:
                    return []
                else:
                    return component.next_component
            elif isinstance(component, Sequence):
                res = []
                for c in component:
                    res = res + get_next_components(c)
                return res
            else:
                raise ValueError(f"'{component}' is not a SimulationComponent or a sequence, which is not allowed.")
        
        def all_None(self, val):
            if val is None:
                return True
            elif isinstance(val, Sequence):
                res = True
                for v in val:
                    res *= all_None(v)
            else:
                return False
        
        self.component_slices = [[first_component]]
        
        while len(self.component_slices[-1]) > 0:
            self.component_slices = self.component_slices + [get_next_components(self.component_slices[-1])]
        
        self.component_slices = self.component_slices[:-1]

        self.runtime_components = [[0.0 for c in sl] for sl in self.component_slices]

    @first_component.setter
    def first_component(self, val):
        if val is None:
            self._first_component = None
            return
        if not isinstance(val, SimulationComponent):
            raise ValueError("First component supplied to Simulation must be derived from SimulationComponent.")
        
        self._first_component = val
        self.make_component_slices(val)

    @property
    def all_components(self):
        return sum(self.component_slices, [])

    @strip_units(observing_time=simulation_units.time, time_resolution=simulation_units.time)
    def run(self, source_list=None, observing_time=None, time_resolution=None, nodes=None, debug=False, print_progress=False):
        time_start = time.time()
        if source_list is not None:
            self.source_list = source_list
        
        if time_resolution is None or observing_time is None:
            if nodes is None:
                self.step(dt=0, debug=debug, print_progress=print_progress)
            else:
                self.parallel_step(dt=0, nodes=nodes, debug=debug, print_progress=print_progress)
            self.evolve_components(1)
        else:
            for t in tqdm(np.arange(0, observing_time, time_resolution)):
                if nodes is None:
                    self.step(dt=time_resolution, debug=debug, print_progress=print_progress)
                else:
                    self.parallel_step(dt=time_resolution, nodes=nodes, debug=debug, print_progress=print_progress)
            self.evolve_components(time_resolution)
        
        # TBD stack current_outputs to datacube if applicable

        time_done = time.time()

        self.runtime_total = time_done - time_start
        self.runtime_overhead = self.runtime_total - sum([sum(self.runtime_components[ii]) for ii in range(len(self.component_slices))])
        return

    @set_units(dt = simulation_units.time)
    def evolve_components(self, dt=None):
        evolve_parameters = {'timestep' : dt}

        for component in self.all_components:
            component.update_simulation(evolve_parameters)
        for component in self.all_components:
            component.evolve(evolve_parameters)
    
    def step(self, source_list=None, dt=None, debug=False, print_progress=False):
        if source_list is None:
            source_list = self.source_list
        
        self.evolve_components(dt)

        for source in source_list:
            iterable = range(len(source.wavelengths))
            if print_progress:
                iterable = tqdm(iterable)
                
            for ii in iterable:
                t0 = time.time()
                self.component_slices[0][0].forward(source, ii) # explicitly pass source to first component
                t1 = time.time()
                self.runtime_components[0][0] += t1-t0
                for jj, sl in enumerate(self.component_slices[1:]):
                    for kk, component in enumerate(sl):
                        if debug:
                            print(component)
                        t0 = time.time()
                        component.apply() # handing over data is done directly by the components
                        t1 = time.time()
                        self.runtime_components[jj+1][kk] += t1 - t0
    
    def parallel_step(self, nodes=4, source_list=None, dt=None, debug=False, print_progress=False):
        def make_mapped_function(source, component_slices):
            # TODO: function must return something, extract info from detector somehow
            def mapped_function(indices):
                for ii in indices:
                    component_slices[0][0].forward(source, ii) # explicitly pass source to first component, each process remakes Pupilgenerator, could be more efficient
                    for jj, sl in enumerate(component_slices[1:]):
                        for kk, component in enumerate(sl):
                            component.apply() # handing over data is done directly by the components
                
                # for now only with hcipyComponent 'Detector' class, TODO: generalize this
                # making a copy might not be necessary, TODO: make sure, because making a copy costs time & memory
                cr = component_slices[-1][0].charge_rate.copy()
                component_slices[-1][0].charge_rate = 0
                
                return cr
            return mapped_function

        if source_list is None:
            source_list = self.source_list
        
        self.evolve_components(dt)

        total_charge_rate = 0
        for source in source_list:
            # make arguments for pool
            index_batches = np.array_split(np.arange(len(source.wavelengths), dtype=int), nodes)
            m = make_mapped_function(source, self.component_slices)

            pool = mp.ProcessPool(nodes = nodes)

            # for now only with hcipyComponent 'Detector' class, TODO: generalize this
            node_charge_rates = pool.map(m, index_batches)
            
            # fixes bug that multiple sources get coadded or some other weird thing when simulating multiple sources
            total_charge_rate += sum(node_charge_rates)
            #self.component_slices[-1][0].charge_rate += sum(node_charge_rates)
        
        self.component_slices[-1][0].charge_rate = total_charge_rate
    
    def statistics(self):
        print(10*"=")
        print(f"Total runtime: {self.runtime_total:.3g} s.")
        print(f"Time spent on overhead: {self.runtime_overhead:.3g} s.")
        print(10*"-")
        print("Time spent per component:")
        for ii, sl in enumerate(self.component_slices):
            for jj, c in enumerate(sl):
                print("\t" + str(c) + f":\t{self.runtime_components[ii][jj]:.3g} s")
        print(10*"=")