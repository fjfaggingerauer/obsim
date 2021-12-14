from .base import hcipyComponent
from ...config import simulation_units
from ...util import strip_units

import numpy as np
import hcipy as hp

__all__ = ['IdealAO']

class IdealAO(hcipyComponent):
    '''
    Removes all phase aberrations that it can possibly correct
    given the number of modes and mode_basis.
    '''

    @strip_units(noise_level=simulation_units.length)
    def __init__(self, num_modes, mode_basis='actuators', noise_level=None):
        self.num_modes = num_modes
        self.mode_basis = mode_basis
        self.noise_level = noise_level

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
        assert prev_component.output_grid_type=='pupil', "Input to Atmosphere should be in pupil grid"
        self.grid = prev_component.output_grid
        self.diameter = 2*np.max(self.grid.x)
        self.transformation_matrix = self.make_transformation_matrix()
        self.inv_transformation_matrix = np.linalg.pinv(self.transformation_matrix,\
            rcond=1e-2)
        self.dm_shape = np.zeros_like(self.grid.x)

    def forward(self, wf):
        sensed_phase = self.atmosphere_opd
        projection = self.inv_transformation_matrix.dot(sensed_phase)
        self.dm_shape = self.transformation_matrix.dot(projection)
        wf_out = wf.copy()
        wf_out.electric_field *= np.exp(-1j*2*np.pi*self.dm_shape/wf.wavelength)
        if self.noise_level is not None:
            phase_noise = self.noise/wf.wavelength 
            wf_out.electric_field *= np.exp(-1j*2*np.pi*phase_noise)
        return wf_out

    def evolve(self, evolve_parameters):
        self.atmosphere_opd = evolve_parameters['atmosphere_opd']
        if self.noise_level is not None:
            self.noise = self.noise_level*np.random.randn(*self.grid.x.shape)

    def update_simulation(self, evolve_parameters):
        self.atmosphere_opd = evolve_parameters['atmosphere_opd']
        if self.noise_level is not None:
            self.noise = self.noise_level*np.random.randn(*self.grid.x.shape)

    def make_transformation_matrix(self):
        if self.mode_basis == 'actuators':
            num_act_across_pupil = int(np.sqrt(self.num_modes))
            self.num_modes = num_act_across_pupil**2
            mu_act = hp.make_pupil_grid(num_act_across_pupil, self.diameter)
            pitch = self.diameter/num_act_across_pupil
            sigma = pitch/np.sqrt(2)
            basis = hp.ModeBasis(hp.make_gaussian_pokes(self.grid, mu_act, sigma))
            transformation_matrix = np.array(basis.transformation_matrix).T

        elif self.mode_basis == 'disk_harmonics':
            basis = hp.mode_basis.make_disk_harmonic_basis(self.grid, \
                        num_modes=self.num_modes, D=self.diameter)
            transformation_matrix = basis.transformation_matrix
        else:
            raise ValueError("Unknown mode basis {0}".format(self.mode_basis))
        return transformation_matrix