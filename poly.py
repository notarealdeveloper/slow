#!/usr/bin/env python3

import fast
import slow

__all__ = [
    'norm',
    'norm_l1',
    'norm_l2',
    'unit',
    'unit_l1',
    'unit_l2',
    'to_row',
    'to_col',
    'dot',
    'proj',
    'dist',
    'cos',
    'breed',
    'mean',
    'nice',
    'coordinates',
    'expand',
    'split',
    'project',
    'reject',
    'explained',
    'unexplained',
    'attention_l1',
    'attention_l2',
    'attention_sm',
    'hardset',
    'softset'
]

def norm(obj):
    t = slow.to_thought(obj)
    return fast.norm(t)

def norm_l1(obj):
    t = slow.to_thought(obj)
    return fast.norm_l1(t)

def norm_l2(obj):
    t = slow.to_thought(obj)
    return fast.norm_l2(t)

def unit(obj):
    t = slow.to_thought(obj)
    return fast.unit(t)

def unit_l1(obj):
    t = slow.to_thought(obj)
    return fast.unit_l1(t)

def unit_l2(obj):
    t = slow.to_thought(obj)
    return fast.unit_l2(t)

def to_row(obj):
    t = slow.to_thought(obj)
    return fast.to_row(t)

def to_col(obj):
    t = slow.to_thought(obj)
    return fast.to_col(t)

def dot(obj1, obj2):
    a = slow.to_thought(obj1)
    b = slow.to_thought(obj2)
    return fast.dot(a, b)

def proj(obj1, obj2):
    a = slow.to_thought(obj1)
    b = slow.to_thought(obj2)
    return fast.proj(a, b)

def dist(obj1, obj2):
    a = slow.to_thought(obj1)
    b = slow.to_thought(obj2)
    return fast.dist(a, b)

def cos(obj1, obj2):
    a = slow.to_thought(obj1)
    b = slow.to_thought(obj2)
    return fast.cos(a, b)

def breed(obj1, obj2):
    a = slow.to_thought(obj1)
    b = slow.to_thought(obj2)
    return fast.breed(a, b)

def mean(objs):
    ts = slow.to_array(objs)
    return fast.mean(ts)

def nice(objs):
    ts = slow.to_array(objs)
    return fast.nice(ts)

def coordinates(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    return fast.coordinates(ts, t)

def expand(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    return fast.expand(ts, t)

def split(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    return fast.split(ts, t)

def project(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    return fast.project(ts, t)

def reject(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    return fast.reject(ts, t)

def explained(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    return fast.explained(ts, t)

def unexplained(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    return fast.unexplained(ts, t)

def attention_l1(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    return fast.attention_l1(ts, t)

def attention_l2(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    return fast.attention_l2(ts, t)

def attention_sm(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    return fast.attention_sm(ts, t)

def hardset(objs, obj, value):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    v = slow.to_thought(value)
    return fast.hardset(ts, t, v)

def softset(objs, obj, value):
    ts = slow.to_array(objs)
    t = slow.to_thought(obj)
    v = slow.to_thought(value)
    return fast.softset(ts, t, v)