from operator import attrgetter
import warnings
from collections.abc import Sequence
import typing as t

import astropy.units as u
# from astropy.units import equivalencies

__all__ = ['make_properties']

# TODO: work out case when '.' in key when setting value, so we have to look
# inside a different object whether the property is set


class PropertyGetter:
    '''
    Helper class that lets users access Property's as a base level attribute
    for the class they are added to. The actual value of a Property 'p' on
    'obj' is at 'obj.properties.p.value', but PropertyGetter lets it be
    accessed as 'obj.p'.

    Based on https://stackoverflow.com/questions/30880842/
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
            # return instance.properties.name.value
            return attrgetter(self.target)(instance)

    def __set__(self, instance, value):
        # split into 'properties.{self.name}' & 'value'
        head, tail = self.target.rsplit('.', 1)

        # get instance.properties.{self.name}
        obj = attrgetter(head)(instance)

        # set instance.properties.{self.name}.value to value
        setattr(obj, tail, value)


class PropertyList:
    '''
    Container object for all Property's added to an object. This object
    facilitates communication between Property's for functions relating
    them. The PropertyList keeps track of Property's related by a function
    and provides attributes which give a quick overview of the current
    state of all Property's in the list.

    Parameters
    ----------
    property_dictionary : dict, optional
        Dictionary used to initialize all properties in the 'make_properties'
        function.
    '''
    def __init__(self, property_dictionary=None):
        self._related_property_sets = []
        self.property_dictionary = property_dictionary

    # TBD what if argument of a function is result of a different function?
    def add_related_property_set(self, properties):
        '''
        Add a set of properties that are not all independent because
        there is a function linking them.
        '''
        property_set = set(properties)
        equal_sets = [s == property_set for s in self._related_property_sets]
        if not any(equal_sets):  # ensure we only add new sets
            self._related_property_sets += [property_set]

    def related_properties(self, name):
        '''
        Get a list of sets of all properties that are related to the given
        property by a function. If z = f(x,y) then this function would return
        [{'x', 'y', 'z'}].
        '''
        return [s for s in self._related_property_sets if name in s]

    @property
    def list(self):
        '''
        All Property's contained in this PropertyList.
        '''
        return list(self.property_dictionary.keys())

    @property
    def values(self):
        '''
        Dictionary of every Property and its current value when evaluated.
        '''
        return {key: getattr(self, key).value for key in self.list}

    @property
    def externally_set(self):
        '''
        Dictionary of every Property and whether it has been externally set.
        '''
        return {key: getattr(self, key).is_set for key in self.list}

    def __str__(self):
        return str(self.values)

    def __getitem__(self, index):
        return getattr(self, index).value

    def __setitem__(self, index, val):
        getattr(self, index).value = val

    def raw_value(self, index):
        '''
        Externally set value of the given Property. Is None if the Property
        has not been externally set.
        '''
        return getattr(self, index).raw_value

    def property_get_value(self, key, forbidden_properties=[]):
        return getattr(self, key).get_value(forbidden_properties)

    def to_dict(self):
        tree = {
            'externally_set': self.externally_set,
            'values': self.values,
        }
        return tree


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
    value
        Value for the parameter.
    unit : astropy.units.Unit or None
        Unit of the property, is None if unitless. Default is None.
    default
        Default value for the parameter. This will be given if no value is
        explicitly set and the property can't be calculated with a function.
        Default is None.
    val_type : object type or None,
        Type of the property. This should only be set if the Property will not
        be a number/array of number with/without units. Will typecheck for
        val_type if it is not None when setting/calculating Property value.
        Overrides unit check if not None. Default is None.
    functions : list of callable
        Functions that can be called to calculate the parameter. Each function
        must be able to be evaluated based on other properties in the parent
        PropertyList. Requires parent to not be None, and function_args to
        contain the call signature for the function.
    function_args : list of list of str
        List of function args per function. Each arg must be given as a str
        of the name of the property to be substituted.
    '''
    def __init__(self, name: str, parent: t.Optional[PropertyList] = None,
                 value=None, unit: t.Optional[u.Unit] = None, default=None,
                 val_type=None, functions: t.Sequence[t.Callable] = [],
                 function_args=[]):

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
        '''
        Default value of this Property.
        '''
        return self._default

    @property
    def unit(self):
        '''
        Unit of this Property. Is None if it has no unit.
        '''
        return self._unit

    @property
    def type(self):
        '''
        Type of this Property. Is None if only a unit is set, in which case
        the type should be astropy.units.Unit
        '''
        return self._type

    @property
    def name(self):
        '''
        str of the name of this Property.
        '''
        return self._name

    @property
    def raw_value(self):
        '''
        Externally set value of this Property. This bypasses the function
        evaluation and default for this Property. If the Property has no
        explicitly set value, this is None. Use the 'value' attribute if you
        want the value for this Property.
        '''
        return self._value

    def reset(self):
        '''
        Reset this Property to its initialization state. This erases the
        externally set value.
        '''
        self.value = None

    def _check(self, val):
        '''
        Makes sure that 'val' has the correct units/is of the correct type.
        Typechecks 'val' against self._type if applicable, and tries to convert
        'val' to self._unit if applicable. None always gets passed through
        this function.
        '''
        if val is None:
            return None

        if self._type is not None:
            if not isinstance(val, self._type):
                raise TypeError(f"Given/calculated value for '{self.name}' \
is of type '{type(val)}', but it must be of type '{self._type}' for this \
Property.")
            return val

        if self.unit is not None:
            # set unit if val is unitless
            if not hasattr(val, 'unit'):
                val = val * self.unit
            else:
                # if units aren't equal try to look if units are equivalent
                if val.unit != self.unit:
                    try:
                        val = val.to(self.unit)
                    except u.UnitsError:
                        raise u.UnitsError(f"Given/calculated value for \
'{self.name}' has unit '{val.unit}', which is incompatible with the expected \
unit '{self.unit}'.")
        else:
            if hasattr(val, 'unit') and val.unit is not None:
                raise u.UnitsError(f"Given/calculated value for '{self.name}' \
has unit '{val.unit}', but unitless is expected.")

        return val

    @property
    def value(self):
        return self.get_value()

    @property
    def is_set(self):
        '''
        Bool whether value of this Property was set externally.
        '''
        return (self._value is not None)

    def get_value(self, forbidden_properties: t.Sequence[str] = []):
        '''
        Get the value of this Property. This is done in the following steps:
        1. If explicitly set, return this value. If not, proceed to:
        2. Try to calculate the value from given functions and other
           Property's. To avoid recursion loops, forbidden properties are
           skipped. If all attempted functions calls are unsuccessful, then:
        3. Return the default value.

        Parameters
        ----------
        forbidden_properties : list of str
            List of Property's (by name) which cannot be used to evaluate
            the current value.

        Returns
        -------
        val
            Value of this Property.
        '''
        # return the set value
        if self.is_set:
            return self._value

        # try to calculate the value from other Property's with the given funcs
        if self._parent is not None:
            for func, desired_args in zip(self._functions, self._func_args):
                # skip iteration if there are any parameters we can't use
                if any([arg in forbidden_properties for arg in desired_args]):
                    continue
                try:
                    args = []
                    for key in desired_args:
                        # if an attribute of an obj, extract the property
                        if '.' in key:
                            parent_obj, tail = key.split('.', 1)
                            instance = self._parent.property_get_value(
                                parent_obj, [self.name] + forbidden_properties)
                            args += [attrgetter(tail)(instance)]
                        else:
                            args += [self._parent.property_get_value(
                                key, [self.name] + forbidden_properties)]

                    # if we found all args we can evaluate func
                    if all([arg is not None for arg in args]):
                        res = func(*args)
                        res = self._check(res)

                        return res
                # if parent_obj is not set the attrgetter will give an
                # AttributeError, meaning we can't find the desired arg,
                # so we can't evaluate the function
                except AttributeError:
                    pass

        # return the default value
        return self._default

    @value.setter
    def value(self, val):
        '''
        Set the value of this Property. This requires checking the type/unit
        of the input, but also if we are trying to set the final parameter in
        a set of Property's linked by a function.

        For example: if z=f(x,y) and x & y are both set, then we can't freely
        set z anymore as this may lead to z != f(x,y). This behaviour is not
        forbidden, but will generate a warning.
        '''
        v = self._check(val)

        # check if we're trying to set all of a set related by a function
        if self._value is None and v is not None and self._parent is not None:
            for related_args in self._parent.related_properties(self.name):
                args = []
                # TBD work out case when '.' in key, so we have to look inside
                # a different object whether the property is set
                for key in related_args:
                    if key == self.name or '.' in key:
                        continue
                    args += [self._parent.raw_value(key)]
                if all([arg is not None for arg in args]):
                    warnings.warn(f"Properties {related_args} have a function \
relating them, but all of them have now been set explicitly. This may lead to \
inconsistencies in relations between these properties.")

        self._value = v


def make_properties(obj, property_list: dict, values: dict = {}):
    '''
    Convert a dictionary with desired properties into a set of attributes
    for and object. The created set of properties allow for a linked set
    of values with defaults and evaluation of unknown properties using a given
    set of functions. When setting a property it may be typechecked or have
    its units enforced.

    Warning: this function must be called at least once in an object of the
    same type as the given object before the property attributes can be
    accessed. This may lead to AttributeErrors if this is not done.

    The property list is a nested dictionary. The top level contains
    name : property settings pairs, where name is a str and property settings
    is another dictionary which may contain the following elements:
    'unit': unit of the property, must be an astropy.units.Unit
    'default': default value for the property is it is not set
    'type': type of the property, when setting the property the new value will
        be typechecked against this
    'functions': a tuple of (function, args) tuples, where function is a
        callable and args a tuple of str containing the names of the other
        properties needed to evaluate the function.

    Parameters
    ----------
    obj
        The object for which to set the properties.
    property_list : dict
        Dictionary containing the information for each property. The structure
        for this dictionary is mentioned above.
    values : dict, optional
        Values to set the properties with. This is equivalent to setting the
        properties manually after they are initialized, but allows for direct
        creation and setting of the properties with this function. The
        dictionary must contain name : value pairs for each property to set.


    Example
    -------
    def add(x,y):
        return x+y

    class A:
        property_list = {
            'x': {'type': float,
                  'default': 1.0}
            'y': {'type': int,
                  'default': 5}
            'z': {'type': float,
                  'functions': ((add, ('x', 'y')),)}
        }
        def __init__(self, **kwargs):
            make_properties(self, self.property_list, kwargs)

    '''
    # add a PropertyList to append properties to
    setattr(obj, 'properties', PropertyList(property_list))

    for key in property_list.keys():
        # set Property defaults
        unit = default = val_type = None
        functions = []
        function_args = []

        # update information we can find
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
                    raise ValueError(f"At {key}: Wrong syntax for adding \
functions to properties. Each function must be provided as \
(function, (*args)).")
                if not callable(element[0]):
                    raise ValueError(f"At {key}: Wrong syntax for adding \
functions to properties. Each function must be provided as \
(function, (*args)).")
                if not isinstance(element[1], Sequence):
                    raise ValueError(f"At {key}: Wrong syntax for adding \
functions to properties. Each function must be provided as \
(function, (*args)).")
                # exception is for attributes of a Property
                for prop in element[1]:
                    if prop not in property_list and '.' not in prop:
                        raise ValueError(f"At {key}: Function argument \
'{prop}' is not a given property, which is necessary for the function to be \
evaluated.")

                functions += [element[0]]
                function_args += [element[1]]
                property_set = [key] + list(element[1])
                obj.properties.add_related_property_set(property_set)

        p = Property(key, obj.properties, None, unit, default,
                     val_type, functions, function_args)

        # make a class attribute for the PropertyGetter if necessary
        if not hasattr(type(obj), key):
            getter = PropertyGetter(key)
            setattr(type(obj), key, getter)

        # set attribute containing the values
        if not hasattr(obj.properties, key):
            setattr(obj.properties, key, p)
        else:
            raise AttributeError(f"Error when setting the property '{key}', \
the property (or one with the same name) was already set.")

    for key in values.keys():
        if key in property_list.keys():
            obj.properties[key] = values[key]
