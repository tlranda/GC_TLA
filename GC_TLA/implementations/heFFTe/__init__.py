from . import heFFTe_problem
def heFFTe_getattr(name):
    return getattr(heFFTe_problem, name)

__getattr__ = heFFTe_getattr
