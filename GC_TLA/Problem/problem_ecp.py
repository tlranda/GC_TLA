# Own library
from GC_TLA.Problem.problem import Problem
from GC_TLA.Utils.param_space import ParamSpace, Real, Integer, Categorical, inf

class ECPProblem(Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
