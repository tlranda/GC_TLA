from . import syr2kgpu_problem
def syr2kgpu_getattr(name):
    return getattr(syr2kgpu_problem, name)

__getattr__ = syr2kgpu_getattr
