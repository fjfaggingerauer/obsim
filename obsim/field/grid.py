import hcipy as hp
import numpy as np
import astropy.units as u

from .coords import UnstructuredCoords

__all__ = ['Grid', 'CartesianGrid']

# TODO: coordinate transformations


class Grid(hp.Grid):
    @classmethod
    def from_dict(cls, tree):
        import copy

        #coords = hp.coordinates.Coords.from_dict(tree['coords'])
        coords = hp.field.coordinates.Coords.from_dict(tree['coords'])
        grid_class = hp.Grid._coordinate_systems[tree['coordinate_system']]

        if 'weights' in tree:
            weights = copy.deepcopy(tree['weights'])
        else:
            weights = None

        return grid_class(coords, weights)

    def to_dict(self):
        tree = {
            'coordinate_system': self._coordinate_system,
            'coords': self.coords.to_dict(),
            'weights': self._weights
        }

        return tree


    def subset(self, criterium):
        if hasattr(criterium, '__call__'):
            indices = criterium(self) != 0
        else:
            indices = criterium

        new_coords = [c[indices] for c in self.coords]

        if self.weigths.size == 1:
            new_weights = self.weights
        else:
            new_weights = self.weights[indices]

        return self.__class__(UnstructuredCoords(new_coords), new_weights)

    @property
    def points(self):
        return np.array([[self.coords[ii][jj] for ii in range(self.ndim)] for jj in range(self.size)], dtype = u.Quantity)

    def axis(self, i):
        return self.coords[i]

    def closest_to(self, p):
        raise NotImplementedError() # might work, but needs testing

    def zeros(self, tensor_shape=None, dtype=None):
        from .field import Field
        if tensor_shape is None:
            shape = [self.size]
        else:
            shape = np.concatenate((tensor_shape, [self.size]))

        return Field(np.zeros(shape, dtype = dtype), self)

    def ones(self, tensor_shape=None, dtype=None):
        from .field import Field
        if tensor_shape is None:
            shape = [self.size]
        else:
            shape = np.concatenate((tensor_shape, [self.size]))

        return Field(np.ones(shape, dtype = dtype), self)

    def empty(self, tensor_shape=None, dtype=None):
        from .field import Field
        if tensor_shape is None:
            shape = [self.size]
        else:
            shape = np.concatenate((tensor_shape, [self.size]))

        return Field(np.empty(shape, dtype = dtype), self)

    @property
    def weights(self):
        if self._weights is None:
            raise NotImplementedError()

        return self._weights

    @weights.setter
    def weights(self, val):
        self._weights = val
    
    @property
    def units(self):
        return self.coords.units
    
    def min(self, axis=None):
        return self.coords.min(axis)
    
    def max(self, axis=None):
        return self.coords.max(axis)

class CartesianGrid(Grid):
    _coordinate_system = 'cartesian_unit'

    def rotate(self, angle, axis=None):
        raise NotImplementedError()

    @staticmethod
    def _get_automatic_weights(coords):
        raise NotImplementedError()