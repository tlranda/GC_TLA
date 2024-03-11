from . import syr2k_problem
def syr2k_getattr(name):
    return getattr(syr2k_problem, name)

__getattr__ = syr2k_getattr
