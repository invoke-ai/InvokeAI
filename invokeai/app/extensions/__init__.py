import os

__all__ = []

dirname = os.path.dirname(os.path.abspath(__file__))

for f in os.listdir(dirname):
    if (os.path.isdir("%s/%s" % (dirname, f))):
        __all__.append(f)