import numpy as np
import astropy.units as u
import hcipy as hp

from .coords import *
from .grid import *

__all__ = ['Field']

# TODO: write out all NotImplemented functions
# TODO: other coordinate systems + grids
# TODO: point weights with units
# TODO: test suite

class Field(u.Quantity, hp.Field):
    def __new__(cls, value, grid):
        obj = u.Quantity(value).view(cls)
        obj.grid = grid

        if hasattr(value, 'unit'): # above line doesn't correctly set unit for some reason
            obj._set_unit(value.unit)
        else:
            obj._set_unit(u.dimensionless_unscaled)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.grid = getattr(obj, 'grid', None)

    def copy(self, order='C'):
        obj = super().copy(order)
        obj._set_unit(self.unit) # above line doesn't set unit for some reason
        obj.grid = self.grid.copy()

        return obj

    @classmethod
    def from_dict(cls, tree):
        unit = u.Unit(tree['unit'])
        val = np.array(tree['values']) * unit
        
        grid = Grid.from_dict(tree['grid'])

        return cls(val, grid)

    def to_dict(self):
        tree = {
            'values' : np.asarray(self),
            'unit' : str(self.unit),
            'grid' : self.grid.to_dict()
        }
        return tree

    def __getstate__(self):
        data_state = u.Quantity(self).__reduce__()[2] # cannot use super() as u.Quantity uses this which causes a recursion loop
        return data_state + (self.grid,)

    def __setstate__(self, state):
        ndarray_state, unit, grid = state

        np.ndarray.__setstate__(self, ndarray_state) # cannot use super() as u.Quantity uses this which causes a recursion loop
        self._unit = unit['_unit']
        self.grid = grid

    def __reduce__(self):
        return (_unitfield_reconstruct,
                (self.__class__, u.Quantity, 0, u.dimensionless_unscaled, 'b'),
                self.__getstate__())

    @property
    def shaped(self):
        raise NotImplementedError()
    
    def at(self, point, method='linear'):
        if self.grid.ndim == 1:
            if method == 'linear':
                grid_points = self.grid.coords[0].value
                
                if isinstance(point, Grid):
                    p = point.coords[0].to_value(self.grid.units[0])
                    interp_value = np.interp(p, grid_points, self.value)
                    return Field(interp_value * self.unit, point)
                else:
                    p = point.to_value(self.grid.units[0])
                    interp_value = np.interp(p, grid_points, self.value)
                    return interp_value * self.unit

            elif method == 'flux_conserving':
                raise NotImplementedError()
        else:
            raise NotImplementedError()


def _unitfield_reconstruct(subtype, baseclass, basevalue, baseunit, basetype):
	'''
    Internal function for building a new Field object for pickling.
    Based on hcipy '_field_reconstruct'.
	
    Parameters
	----------
	subtype
		The class of Field.
	baseclass
		The array class that was used for the Field.
	basevalue
		The default value for baseclass.
    basevalue
		The default unit for baseclass.
	basetype
		The data type of the Field.
	Returns
	-------
	Field
		The built Field object.
	'''
	data = u.Quantity.__new__(baseclass, basevalue, baseunit, basetype)
	grid = None

	return subtype.__new__(subtype, data, grid)

hp.field.coordinates.Coords._add_coordinate_type('separated_unit', SeparatedCoords)
hp.field.grid.Grid._add_coordinate_system('cartesian_unit', CartesianGrid)