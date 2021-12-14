from ..util import set_units
from ..config import default_units

__all__ = ['Model', 'ModelSum']

class ModelSum(object):
    def __init__(self, models):
        self.models = models
    
    def initialise_for(self, source):
        self.source = source

        for model in self.models:
            model.initialise_for(source)
    
    def __call__(self, wavelengths):
        output = [model(wavelengths) for model in self.models]
        return sum(output)
    
    def __add__(self, m):
        if not isinstance(m, Model) and not isinstance(m, ModelSum):
            raise ValueError(f"Addition not supported between a Model and {type(m)}.")
        
        if isinstance(m, ModelSum):
            self.models = self.models + m.models
        else:
            self.models = self.models + [m]
        
        return self
    
    def __radd__(self, m):
        return self.__add__(m)

class Model(object):
    def __init__(self):
        pass
    
    def initialise_for(self, source):
        self.source = source

    def at(self, wavelengths):
        raise NotImplementedError

    @set_units(wavelength=default_units.length)
    def __call__(self, wavelengths):
        raise NotImplementedError
    
    def __add__(self, m):
        if not isinstance(m, Model) and not isinstance(m, ModelSum):
            raise ValueError(f"Addition not supported between Model and {type(m)}.")

        if isinstance(m, ModelSum):
            return m + self
        else:
            return ModelSum([self, m])
    
    def __radd__(self, m):
        return self.__add__(m)