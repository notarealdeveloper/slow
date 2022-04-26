#!/usr/bin/env python3

""" Strong typing. """

___all__ = [
    'to_thought',
    'to_array',
    'is_instance',
]

import jax
import jaxlib
import jax.numpy as jnp

import types
import typing
from _collections_abc import dict_values

def to_thought(arg):
    if isinstance(arg, jaxlib.xla_extension.DeviceArray):
        return arg
    if isinstance(arg, jax.core.Tracer):
        return arg
    if hasattr(arg, 'think'):
        return arg.think()
    raise TypeError(f"Cannot coerce to thought: {arg!r}")


def to_array(arg):
    if isinstance(arg, jaxlib.xla_extension.DeviceArray):
        return arg
    if isinstance(arg, jax.core.Tracer):
        return arg
    if isinstance(arg, dict):
        arg = list(arg.values())
    if isinstance(arg, dict_values):
        arg = list(arg)
    if isinstance(arg, (list, tuple, set)):
        thoughts = [to_thought(o) for o in arg]
        return jnp.stack(thoughts, axis=0)
    if hasattr(arg, '__array__'):
        return arg.__array__()
    raise TypeError(f"Cannot coerce to array: {arg!r}")


def is_instance(obj, cls):

    """ Turducken typing. """

    if isinstance(cls, tuple):
        for scls in cls:
            if is_instance(obj, scls):
                return True
        else:
            return False

    if type(cls) is typing._UnionGenericAlias \
    and cls.__origin__ is typing.Union:
        return is_instance(obj, cls.__args__)

    if not isinstance(cls, types.GenericAlias):
        return isinstance(obj, cls)

    if not is_instance(obj, cls.__origin__):
        return False

    ocls = cls.__origin__
    args = cls.__args__

    if ocls is list:
        assert len(args) == 1
        itemcls = args[0]
        for item in obj:
            if not is_instance(item, itemcls):
                return False
        return True
    if ocls is dict:
        assert len(args) == 2
        keycls, valcls = args
        for key, val in obj.items():
            if not is_instance(key, keycls):
                return False
            if not is_instance(val, valcls):
                return False
        return True
    if ocls is tuple:
        for sobj, scls in zip(obj, args):
            if not is_instance(sobj, scls):
                return False
        return True

    raise TypeError(obj, cls)

