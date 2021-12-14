import hcipy as hp
import numpy as np
import astropy.units as u

__all__ = ['UnstructuredCoords', 'SeparatedCoords']

# TODO: write out all NotImplemented functions

class UnstructuredCoords(hp.UnstructuredCoords):
    def __init__(self, coords):
        try:
            self.units = [c[0].unit if (hasattr(c[0], 'unit') and c[0].unit) else u.dimensionless_unscaled for c in coords]
            self.coords = [np.array(c).astype(float) for c in coords]
        except TypeError: # if there's only 1 axis the above lines fail
            self.units = [coords[0].unit]
            self.coords = [np.array(coords).astype(float)]
        

    @classmethod
    def from_dict(cls, tree):
        if tree['type'] != 'unstructured_unit':
            raise ValueError('The type of coordinates should be "unstructured".')

        inp = [c * u.Unit(unit) for c, unit in zip(tree['coords'], tree['units'])]

        return cls(inp)

    def to_dict(self):
        tree = {
            'type': 'unstructured_unit',
            'units' : [str(u) for u in self.units],
            'coords': self.coords
        }

        return tree

    def __getitem__(self, i):
        val = super().__getitem__(i)

        return val * self.units[i]

    def __iadd__(self, b):
        b_unitless = np.zeros(len(b))
        for ii in range(len(b)):
            if hasattr(b[ii], 'unit'):
                b_unitless[ii] = b[ii].to(self.units[ii]).value
            elif self.units[ii] == u.dimensionless_unscaled:
                b_unitless[ii] = b[ii]
            else:
                raise u.UnitsError()

        super().__iadd__(b_unitless)

        return self


    def __imul__(self, f):
        if np.isscalar(f):
            f_unitless = f
        elif hasattr(f, 'unit') and f.size == 1:
            self.units = [u * f.unit for u in self.units]
            f_unitless = f.value
        else:
            f_unitless = np.zeros(len(f))
            for ii in range(len(f)):
                if hasattr(f[ii], 'unit'):
                    self.units[ii] *= f[ii].unit
                    f_unitless[ii] = f[ii].value
                else:
                    f_unitless[ii] = f[ii]

        super().__imul__(f_unitless)

        return self
    
    def min(self, axis=None):
        raise NotImplementedError()
    
    def max(self, axis=None):
        raise NotImplementedError()




class SeparatedCoords(hp.SeparatedCoords):
    def __init__(self, separated_coords):
        try:
            separated_coords[0][0]
            coords = [u.Quantity(q) for q in separated_coords]
        except TypeError: # error if a single Quantity is given should be "'Quantity' object with a scalar value does not support indexing"
            coords = [u.Quantity(separated_coords)]
        
        self.units = [q.unit if q.unit else u.dimensionless_unscaled for q in coords]
        self.separated_coords = [q.value for q in coords]

    @classmethod
    def from_dict(cls, tree):
        if tree['type'] != 'separated_unit':
            raise ValueError('The type of coordinates should be "separated_units".')

        inp = [val * u.Unit(unit) for val, unit in zip(tree['separated_coords'], tree['units'])]
        return cls(inp)

    def to_dict(self):
        tree = {
            'type': 'separated_unit',
            'separated_coords': self.separated_coords,
            'units' : [str(u) for u in self.units]
        }

        return tree

    def __getitem__(self, i):
        val = super().__getitem__(i)

        return val * self.units[i]

    def __iadd__(self, b):
        try:
            if np.isscalar(b) or b.size == 1:
                b = np.tile(b, len(self.units))
        except AttributeError:
            pass

        b_unitless = np.zeros(len(b))

        for ii in range(len(b)):
            if hasattr(b[ii], 'unit'):
                b_unitless[ii] = b[ii].to(self.units[ii]).value
            elif self.units[ii] == u.dimensionless_unscaled:
                b_unitless[ii] = b[ii]
            else:
                raise u.UnitsError()

        super().__iadd__(b_unitless)

        return self


    def __imul__(self, f):
        if np.isscalar(f):
            f_unitless = f
        elif hasattr(f, 'unit') and f.size == 1:
            self.units = [u * f.unit for u in self.units]
            f_unitless = f.value
        else:
            f_unitless = np.zeros(len(f))
            for ii in range(len(f)):
                if hasattr(f[ii], 'unit'):
                    self.units[ii] *= f[ii].unit
                    f_unitless[ii] = f[ii].value
                else:
                    f_unitless[ii] = f[ii]

        super().__imul__(f_unitless)

        return self
    
    def min(self, axis=None):
        minima = [q.min()*u for q, u in zip(self.separated_coords, self.units)]
        if axis is not None:
            return minima[axis]
        else:
            return minima
    
    def max(self, axis=None):
        maxima = [q.max()*u for q, u in zip(self.separated_coords, self.units)]
        if axis is not None:
            return maxima[axis]
        else:
            return maxima