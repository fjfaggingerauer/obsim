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
        for par in wrapped_signature.parameters.values():
            if par.name in bound_args.arguments:  # passed explicitly
                arg = bound_args.arguments[par.name]
            else:  # if not provided use default
                arg = par.default
            # attach unit if needed
            if par.name in kwargs and not hasattr(arg, 'unit') \
                    and arg is not None:
                unit = kwargs[par.name]
                if isinstance(kwargs[par.name], Sequence):
                    unit = unit[0]
                arg = arg * unit

            if par.name in kwargs and arg is None and par.default is not None:
                raise ValueError(f"Passing '{par.name}' with a value of None\
 is not allowed when it must have units and its default value is not None.")

            new_input[par.name] = arg

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
            if param.name in bound_args.arguments:  # passed explicitly
                arg = bound_args.arguments[param.name]
            else:  # if not provided use default
                arg = param.default
            # attach unit if needed
            if param.name in kwargs:
                if hasattr(arg, 'unit'):
                    arg = arg.to(kwargs[param.name]).value
                elif arg is not None:
                    raise ValueError("Couldn't strip unit off of '{0}' because\
                         it has no unit.".format(param.name))

            new_input[param.name] = arg

        return func(**new_input)
    return f


class UnitChecker:
    @classmethod
    def as_decorator(cls, func=None, **kwargs):
        '''
        Extends the usage of astropy.unit.quantity_input by not only checking
        the units of the function input and generating an error if they are
        inconsistent with the set units, but also attaches the required units
        to an input if it is dimensionless.

        Syntax is the same as astropy.unit.quantity_input
        '''
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
        '''
        Extends the usage of astropy.unit.quantity_input by removing the
        supplied units from the input after checking them, meaning a set of
        floats is supplied to the underlying function.

        Syntax is the same as astropy.unit.quantity_input
        '''
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
