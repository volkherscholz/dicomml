import sys


def resolve(s: str, prefix: str = 'dicomml'):
    """
    Resolve strings to objects using standard import and attribute
    syntax.
    -- copied from lib/logging/config
    """
    if prefix == 'tf':
        prefix = 'tensorflow'
    name = '{prefix}.{s}'.format(prefix=prefix, s=s).split('.')
    used = name.pop(0)
    try:
        found = __import__(used)
        for frag in name:
            used += '.' + frag
            try:
                found = getattr(found, frag)
            except AttributeError:
                __import__(used)
                found = getattr(found, frag)
        return found
    except ImportError:
        e, tb = sys.exc_info()[1:]
        v = ValueError('Cannot resolve %r: %s' % (s, e))
        v.__cause__, v.__traceback__ = e, tb
        raise v
