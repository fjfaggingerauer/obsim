__all__ = ['set_units', 'strip_units']

import astropy.units as u
from collections.abc import Sequence
import inspect
from functools import wraps

_default_equivalencies = u.temperature()


def _make_single_units(**kwargs):
    for val in kwargs.keys():
        # check if unit is unambiguous
        if isinstance(kwargs[val], Sequence):
            if len(kwargs[val]) > 1:
                raise ValueError("Unit of `{0}` is ambiguous.".format(val))
            kwargs[val] = kwargs[val][0]

def _add_default_units(func, **kwargs):
    wrapped_signature = inspect.signature(func)

    @wraps(func)
    def f(*func_args, **func_kwargs):
        # get provided arguments
        bound_args = wrapped_signature.bind(*func_args, **func_kwargs)
        new_input = {}

        # attach default units to every parameter where applicable
        for param in wrapped_signature.parameters.values():
            if param.name in bound_args.arguments: # passed explicitly
                arg = bound_args.arguments[param.name]
            else: # if not provided use default
                arg = param.default
            # attach unit if needed    
            if param.name in kwargs and not hasattr(arg, 'unit') and arg is not None:
                unit = kwargs[param.name]
                if isinstance(kwargs[param.name], Sequence):
                    unit = unit[0]
                arg = arg * unit
            
            if param.name in kwargs and arg is None and param.default is not None:
                raise ValueError(f"Passing '{param.name}' with a value of None is not allowed when it must have units and its default value is not None.")
            
            new_input[param.name] = arg
        
        return func(**new_input)
    
    return f

def _strip_units(func, **kwargs):
    wrapped_signature = inspect.signature(func)

    @wraps(func)
    def f(*func_args, **func_kwargs):
        # get provided arguments
        bound_args = wrapped_signature.bind(*func_args, **func_kwargs)
        new_input = {}

        # attach default units to every parameter where applicable
        for param in wrapped_signature.parameters.values():
            if param.name in bound_args.arguments: # passed explicitly
                arg = bound_args.arguments[param.name]
            else: # if not provided use default
                arg = param.default
            # attach unit if needed    
            if param.name in kwargs:
                if hasattr(arg, 'unit'):
                    arg = arg.to(kwargs[param.name]).value
                elif arg is not None:
                    raise ValueError("Couldn't strip unit off of '{0}' because it has no unit.".format(param.name))
            
            new_input[param.name] = arg
        
        return func(**new_input)
    return f

class UnitChecker:
    @classmethod
    def as_decorator(cls, func=None, **kwargs):
        self = cls(**kwargs)
        if func is not None and not kwargs:
            return self(func)
        else:
            return self
    
    def __init__(self, func=None, **kwargs):
        self.kwargs = kwargs
        self.kwargs['equivalencies'] = _default_equivalencies

    def __call__(self, func):
        check_units = u.quantity_input(**self.kwargs)

        return _add_default_units(check_units(func), **self.kwargs)

class UnitStripper:
    @classmethod
    def as_decorator(cls, func=None, **kwargs):
        self = cls(**kwargs)
        if func is not None and not kwargs:
            return self(func)
        else:
            return self
    
    def __init__(self, func=None, **kwargs):
        _make_single_units(**kwargs)
        self.kwargs = kwargs
        self.kwargs['equivalencies'] = _default_equivalencies

    def __call__(self, func):
        check_units = u.quantity_input(**self.kwargs)

        return check_units(_strip_units(func, **self.kwargs))

set_units = UnitChecker.as_decorator
strip_units = UnitStripper.as_decorator