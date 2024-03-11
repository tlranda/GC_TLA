from . import embedded_problem
def embedded_getattr(name):
    return getattr(embedded_problem, name)

__getattr__ = embedded_getattr
