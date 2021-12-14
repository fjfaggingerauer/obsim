from operator import attrgetter
import warnings
from collections.abc import Sequence

import astropy.units as u
#from astropy.units import equivalencies

__all__ = ['make_properties', 'PropertyList', 'Property']

# TODO: work out case when '.' in key when setting value, so we have to look inside a different object whether the property is set

# based on https://stackoverflow.com/questions/30880842/dynamically-creating-attribute-setter-methods-for-all-properties-in-class-pyth
class PropertyGetter:
    '''
    Helper class that lets users access Property's as a base level attribute
    for the class they are added to. The actual value of a Property 'p' on 
    'obj' is at 'obj.properties.p.value', but PropertyGetter lets it be 
    accessed as 'obj.p'.
    '''
    def __init__(self, name):
        self.name = name

    @property
    def target(self):
        return 'properties.' + self.name + '.value'
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return attrgetter(self.target)(instance) # return instance.properties.name.value
    
    def __set__(self, instance, value):
        head, tail = self.target.rsplit('.', 1) # split into 'properties.{self.name}' & 'value'
        obj = attrgetter(head)(instance) # get instance.properties.{self.name}
        setattr(obj, tail, value) # set instance.properties.{self.name}.value to value

class Property:
    '''
    Base class for a single property. Allows for properties with/without units,
    supporting a default and evaluation based on other properties. Any values
    automatically have units added if required.

    Parameters
    ----------
    name : str
        Property name.
    parent : PropertyList or None
        List that the property belongs to. This must be given if you want to
        be able to evaluate a property with a function. Default is None.
    value : astropy.units.Quantity or ndarray or scalar
        Value for the parameter. Must be able to be converted to an astropy
        Quantity.
    unit : astropy.units.Unit or None
        Unit of the property, is None if unitless. Default is None.
    default : astropy.units.Quantity or ndarray or scalar
        Default value for the parameter. This will be given if no value is 
        explicitly set and the property can't be calculated with a function.
    val_type : object type or None,
        Type of the property. This should only be set if the Property will not be
        a number/array of number with/without units. Will typecheck for val_type 
        if it is not None when setting/calculating Property value. Overrides unit 
        check if not None. Default is None.
    functions : list of callable
        Functions that can be called to calculate the parameter. Each function
        must be able to be evaluated based on other properties in the parent
        PropertyList. Requires parent to not be None, and function_args to 
        contain the call signature for the function.
    function_args : list of list of str
        List of function args per function. Each arg must be given as a str
        of the name of the property to be substituted.
    '''
    def __init__(self, name, parent=None, value=None, unit=None, default=None, val_type=None, functions=[], function_args=[]):
        
        self._unit = unit
        self._functions = functions
        self._func_args = function_args
        self._parent = parent
        self._name = name
        self._value = None
        self._type = val_type
        self._default = self._check(default)
        
        self.value = value

    @property
    def default(self):
        return self._default
    
    @property
    def unit(self):
        return self._unit
    
    @property
    def name(self):
        return self._name

    @property
    def raw_value(self):
        return self._value
    
    def reset(self):
        self.value = None
    
    def _check(self, val):
        '''
        Makes sure that 'val' has the correct units/is of the correct type. Typechecks
        'val' against self._type if applicable, and tries to convert 'val' to self._unit
        if applicable. None always gets passed through this function.
        '''
        if val is None:
            return None

        if self._type is not None:
            if not isinstance(val, self._type):
                raise ValueError(f"Given/calculated value for '{self.name}' is of type '{type(val)}', but it must be of type '{self._type}' for this Property.")
            return val

        if self.unit is not None:
            if not hasattr(val, 'unit'):
                #raise u.UnitsError(f"Given/calculated value for {self.name} is unitless, but unit {self.unit} is expected.")
                val = val * self.unit
            else:
                if val.unit != self.unit: # first quick check if units are directly equal, if not try to look if units are equivalent
                    try: 
                        val = val.to(self.unit)
                    except u.UnitsError:
                        raise u.UnitsError(f"Given/calculated value for '{self.name}' has unit '{val.unit}', which is incompatible with the expected unit '{self.unit}'.")
        else:
            if hasattr(val, 'unit') and val.unit is not None:
                raise u.UnitsError(f"Given/calculated value for '{self.name}' has unit '{val.unit}', but unitless is expected.")
        
        return val
        
    @property
    def value(self):
        return self.get_value()
    
    @property
    def is_set(self):
        return (self._value is not None)

    def get_value(self, forbidden_properties=[]):
        # first try to return _value, this only happens if value was explicitly set
        if self.is_set:
            return self._value
        
        # secondly try to calculate it from other values with the given funcs
        if self._parent is not None:
            for func, desired_args in zip(self._functions, self._func_args):
                # ensure we don't end in a loop by recursively trying to calculate the same set of values
                if any([arg in forbidden_properties for arg in desired_args]): 
                    continue
                try:
                    args = []
                    for key in desired_args:
                        if '.' in key:
                            parent_obj, tail = key.split('.', 1) # split parent object and desired property
                            instance = self._parent.property_get_value(parent_obj, [self.name] + forbidden_properties) # get the parent object
                            args += [attrgetter(tail)(instance)] # get the appropriate property
                        else:
                            args += [self._parent.property_get_value(key, [self.name] + forbidden_properties)]
                    #args = [self._parent[key] for key in desired_args]
                    if all([arg is not None for arg in args]):
                        res = func(*args)
                        res = self._check(res)
                            
                        return res
                except AttributeError: # if parent_obj is not set the attrgetter will give an AttributeError, meaning we can't find the desired arg, so we can't evaluate the function
                    pass
        
        # finally return a default value
        return self._default

    @value.setter
    def value(self, val):
        val = self._check(val)

        # check if we're trying to set all values of a set related by a function
        if self._value is None and val is not None and self._parent is not None:
            for related_args in self._parent.related_properties(self.name):
                args = []
                for key in related_args:
                    if key == self.name or '.' in key: # TBD work out case when '.' in key, so we have to look inside a different object whether the property is set
                        continue
                    args += [self._parent._raw_value(key)]
                #args = [self._parent._raw_value(key) for key in related_args if key != self.name]
                if all([arg is not None for arg in args]): 
                    warnings.warn(f"Properties {related_args} have a function relating them, but all of them have now been set explicitly. This may lead to inconsistencies in relations between these properties.")
            
        self._value = val
    


# simple class to which Property's are added as attributes
class PropertyList:
    def __init__(self, property_dictionary=None):
        self._related_property_sets = []
        self.property_dictionary = property_dictionary
    
    # TBD think about when argument of a function is result of a different function
    def add_related_property_set(self, properties):
        property_set = set(properties)
        equal_sets = [s == property_set for s in self._related_property_sets]
        if not any(equal_sets): # ensure we only add new sets
            self._related_property_sets += [property_set]
    
    def related_properties(self, name):
        '''
        Get a list of sets of all properties that are related to the given
        property by a function. If z = f(x,y) then this function would return
        related_properties(z) = [{'x', 'y', 'z'}].
        '''
        return [s for s in self._related_property_sets if name in s]
    
    @property
    def list(self):
        '''
        All Property's contained in this PropertyList.
        '''
        #all_props = list(self.__dict__.keys())
        #return [x for x in all_props if not x.startswith('_')]
        return list(self.property_dictionary.keys())

    @property
    def values(self):
        '''
        Dictionary of every Property and its current value when evaluated.
        '''
        return {key: getattr(self, key).value for key in self.list}
    
    @property
    def externally_set(self):
        return {key: getattr(self, key).is_set for key in self.list}
        #return {key: (getattr(self, key)._value is not None) for key in self.list}
    
    def __str__(self):
        return str(self.values)
    
    def __getitem__(self, index):
        return getattr(self, index).value
    
    def __setitem__(self, index, val):
        getattr(self, index).value = val
    
    def _raw_value(self, index):
        return getattr(self, index).raw_value
    
    def property_get_value(self, key, forbidden_properties=[]):
        return getattr(self, key).get_value(forbidden_properties)

    def to_dict(self):
        tree = {
            'externally_set' : self.externally_set,
            'values' : self.values,
        }
        return tree



def make_properties(obj, property_list, values):
    # add a property list to append properties to
    setattr(obj, 'properties', PropertyList(property_list))

    for key in property_list.keys():
        # build Property
        unit = default = val_type = None
        functions = []
        function_args = []

        if 'unit' in property_list[key].keys():
            unit = property_list[key]['unit']
        if 'default' in property_list[key].keys():
            default = property_list[key]['default']
        if 'type' in property_list[key].keys():
            val_type = property_list[key]['type']
        
        if 'functions' in property_list[key].keys():
            function_list = property_list[key]['functions']
            for element in function_list:
                # ensure input is as expected
                if len(element) != 2:
                    raise ValueError(f"At {key}: Wrong syntax for adding functions to properties. Each function must be provided as (function, (*args)).")
                if not callable(element[0]):
                    raise ValueError(f"At {key}: Wrong syntax for adding functions to properties. Each function must be provided as (function, (*args)).")
                if not isinstance(element[1], Sequence):
                    raise ValueError(f"At {key}: Wrong syntax for adding functions to properties. Each function must be provided as (function, (*args)).")
                if not all([(prop in property_list.keys()) for prop in element[1] if not '.' in prop]): # exception is for attributes of a Property
                    for x in element[1]:
                        if x not in property_list.keys():
                            raise ValueError(f"At {key}: Function argument {x} is not a given property, which is necessary for the function to be evaluated.")
                
                functions += [element[0]]
                function_args += [element[1]]
                obj.properties.add_related_property_set([key] + list(element[1]))

        p = Property(key, obj.properties, None, unit, default, val_type, functions, function_args)
        
        # let users directly access Property from obj
        getter = PropertyGetter(key)

        # add attributes to object
        if not hasattr(type(obj), key):
            setattr(type(obj), key, getter) # make a class attribute for the PropertyGetter
        setattr(obj.properties, key, p) # set attribute containing the values
    
    for key in values.keys():
        if key in property_list.keys():
            obj.properties[key] = values[key]