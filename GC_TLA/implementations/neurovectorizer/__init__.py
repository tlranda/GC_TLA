from . import neurovectorizer_problem
def neurovectorizer_getattr(name):
    return getattr(neurovectorizer_problem, name)

__getattr__ = neurovectorizer_getattr

