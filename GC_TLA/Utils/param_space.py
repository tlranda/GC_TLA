"""
    Simple container class taken from discontinued autotune library by Victor Zhong (Formerly https://github.com/vzhong/autotune)
"""

from skopt.utils import use_named_args
# More feature-complete class
from skopt.space import Space
# Useful and compatible types to form dimensions from
from skopt.space import Real, Integer, Categorical
# Useful for numeric spaces
from numpy import inf

class ParamSpace(Space):
    def to_dict(self, params_list):
        """
            While self.dimensions should not be mutated after intialization, it is not available at
            class-instantiation time. The decorator is useful for integrating with some other works,
            so leave this code alone
        """
        @use_named_args(self.dimensions)
        def to_params_dict(**params):
            return params
        return to_params_dict(params_list)
