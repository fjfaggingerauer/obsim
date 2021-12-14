from ...config import simulation_units
from ...util import strip_units
from ...sources import Background
from .base import hcipyComponent

import astropy.units as u
import astropy.coordinates as c
import numpy as np
import hcipy as hp

__all__ = ['hcipyVacuumPupil', 'hcipyBackground', 'hcipyPupilGenerator']

class hcipyVacuumPupil(object):
    @strip_units(wavelengths = simulation_units.length, spectrum = simulation_units.flux_wavelength_density)
    def __init__(self, amplitude=None, opd=None, wavelengths=None, spectrum=None, pointing=None):
        self.amplitude = amplitude
        self.opd = opd
        self.wavelengths = wavelengths
        self.spectrum = spectrum
        self.pointing = pointing
    
    def __len__(self):
        try:
            return len(self.wavelengths)
        except TypeError:
            return 0
    
    @property
    def is_empty(self):
        return (self.amplitude is None)
    
    @property
    def wavelength_bin_sizes(self):
        #if len(self.wavelengths == 1):
        #    return [1]
        dwl0 = self.wavelengths[0] - (self.wavelengths[1]-self.wavelengths[0])
        dlams = np.diff(self.wavelengths,prepend=dwl0)
        return dlams

    @property
    def wavenumbers(self):
        return 2*np.pi/self.wavelengths
    
    def make_pupil(self, index):
        if self.is_empty:
            return None
        
        wavelength = self.wavelengths[index]
        
        amplitude = self.amplitude * np.sqrt(self.spectrum[index] * self.wavelength_bin_sizes[index])
        phase = self.opd * self.wavenumbers[index]
        E = amplitude * np.exp(1j*phase)

        return hp.Wavefront(E, wavelength)
    
    def __getitem__(self, index):
        return self.make_pupil(index)
    
    def __iter__(self):
        self._current_index = 0
        return self
    
    def __next__(self):
        self._current_index += 1
        if self._current_index > len(self.wavelengths):
            raise StopIteration
        return self[self._current_index-1]

class hcipyBackground(object):
    @strip_units(wavelengths = simulation_units.length, spectrum = simulation_units.flux_wavelength_density)
    def __init__(self, wavelengths=[], spectrum=None):
        self.wavelengths = wavelengths
        self.values = spectrum.copy() * self.wavelength_bin_sizes

        self.eval_index = None
    
    def __len__(self):
        return len(self.wavelengths)
    
    @property
    def is_empty(self):
        return (len(self.wavelengths) == 0)
    
    @property
    def pointing(self):
        return None
    
    @property
    def wavelength_bin_sizes(self):
        dwl0 = self.wavelengths[0] - (self.wavelengths[1]-self.wavelengths[0])
        dlams = np.diff(self.wavelengths, prepend=dwl0)
        return dlams

    @property
    def wavenumbers(self):
        return 2*np.pi/self.wavelengths

    @property
    def value(self):
        if self.eval_index is None:
            raise AttributeError("Index must first be set before accessing a value.")
        
        return self.values[self.eval_index]
    
    @value.setter
    def value(self, x):
        if self.eval_index is None:
            raise AttributeError("Index must first be set before accessing a value.")
        
        self.values[self.eval_index] = x
    
    @property
    def wavelength(self):
        if self.eval_index is None:
            raise AttributeError("Index must first be set before accessing a wavelength.")

        return self.wavelengths[self.eval_index]
    
    def set_index(self, index):
        if self.is_empty:
            return None
        
        self.eval_index = index
        
        return self 
    
    def __getitem__(self, index):
        return self.set_index(index)
    
    def __iter__(self):
        self._current_index = 0
        return self
    
    def __next__(self):
        self._current_index += 1
        if self._current_index > len(self.wavelengths):
            raise StopIteration
        return self[self._current_index-1]
    
    def copy(self):
        output = hcipyBackground(self.wavelengths*simulation_units.length, self.values*simulation_units.flux_wavelength_density/self.wavelength_bin_sizes)
        output.eval_index = self.eval_index

        return output


class hcipyPupilGenerator(hcipyComponent):
    def __init__(self, telescope, max_source_number=128):
        self.telescope = telescope
        self.max_source_number = max_source_number
        self.sources = []
        self._current_source = (None, None)
    
    @property
    def input_grid(self):
        return None

    @property
    def output_grid(self):
        return self.grid
    
    @property
    def input_grid_type(self):
        return None

    @property
    def output_grid_type(self):
        return 'pupil'
    
    def initialise_for(self, component):
        raise ValueError("InputPupilGenerator cannot have a component in front of it.")

    @property
    def telescope(self):
        return self._telescope
    
    @telescope.setter
    def telescope(self, val):
        self._telescope = val

        #self.pointing = val.pointing
        #self.grid = val.pupil_grid
    
    @property
    def pointing(self):
        return self.telescope.pointing
    
    @property
    def grid(self):
        return self.telescope.pupil_grid

    def forward(self, source, ii):
        # check if we're still using the same source as in previous calls
        if source is self._current_source[0]:
            self.output = self._current_source[1][ii]
            return self.output

        # see if we've made a VacuumPupil already for the current source
        for ii, (s, p) in enumerate(self.sources):
            if source is s: # want same object in this case, so no '=='
                if not np.all(p.pointing == self.pointing):
                    p = self.make_new_vacuum_pupil(source)
                    self.sources[ii] = (source, p)

                self._current_source = (source, p)
                self.output = p[ii]
                return self.output
        
        # source not yet in list, so add it
        p = self.make_new_vacuum_pupil(source)
        self.sources += [(source, p)]
        
        # remove oldest entry if self.sources is getting too long
        if len(self.sources) > self.max_source_number:
            self.sources.pop(0)

        self._current_source = (source, p)
        self.output = p[ii]
        return self.output
    
    def make_new_vacuum_pupil(self, source, max_offset=None):
        pupil_grid = self.grid

        # make hcipyBackground if required
        if isinstance(source, Background):
            return hcipyBackground(source.wavelengths, source.spectrum)

        # only point sources are supported for now
        if source.extent is not None:
            raise NotImplementedError

        angular_offset = self.calculate_angular_offset(source.location)

        if max_offset is None:
            # calculate max offset by what could possibly end up in a focal plane
            ft_grid = hp.make_fft_grid(pupil_grid)
            max_k_offset = np.max(np.linalg.norm(ft_grid.points, axis=1)) / simulation_units.length
            max_offset = np.arctan(max_k_offset/min(source.wavenumbers)) # automatically has units of rad

        # make empty PupilGenerator if object won't appear in focal plane
        if np.any(angular_offset > max_offset):
            vacuum_pupil = hcipyVacuumPupil()
        else:
            amplitude = pupil_grid.ones()
            opd = hp.Field((pupil_grid.points @ angular_offset.to(u.rad).value), pupil_grid)
            
            vacuum_pupil = hcipyVacuumPupil(amplitude, opd, source.wavelengths, source.spectrum)
        
        return vacuum_pupil
    
    def calculate_angular_offset(self, object_location):
        if isinstance(object_location, u.Quantity):
            return object_location - self.pointing

        elif isinstance(object_location, c.SkyCoord):
            if self.telescope.location is None:
                raise ValueError("When a source has a physical location, the telescope must have this as well.")
            if self.telescope.observation_time is None:
                raise ValueError("When a source has a physical location, an observation time must be given.")
            if self.telescope.physical_pointing is None: # make sure telescope pointing is an AltAz instance
                self.telescope.physical_pointing = c.SkyCoord(c.AltAz(alt=self.pointing[0], az=self.pointing[1], obstime = self.telescope.observation_time, location = self.telescope.location))

            telescope_frame = c.SkyOffsetFrame(origin=self.telescope.physical_pointing)
            altaz_source = object_location.transform_to(c.AltAz(obstime = self.telescope.observation_time, location = self.telescope.location))

            offset = altaz_source.transform_to(telescope_frame)

            if isinstance(offset.data, u.Quantity):
                return offset.data
            elif isinstance(offset.data, c.UnitSphericalRepresentation): 
                # this gets returned if the offset is so large that spherical effects are important
                # this will typically mean that the source is far outside the telescope fov
                return u.Quantity([offset.data.lon, offset.data.lat])
            else:
                raise RuntimeError(f"Encountered an unknown output when calculating angular offset of a source, received a {type(offset.data)}.")


            # TODO: convert altaz to correct offset for pointing (requires telescope rotation I think)

    
    def evolve(self, evolve_parameters):
        for s, p in self.sources:
            s.evolve(evolve_parameters)
            #evolve_parameters['airmass'] = 