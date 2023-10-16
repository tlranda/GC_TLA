"""
Benchmarks are here to help you reproduce our results.
They can also be used to test your installation of ytopt or to discover the many parameters of a search.
"""

"""
Benchmarks are here to help you reproduce our results.
They can also be used to test your installation of ytopt or to discover the many parameters of a search.
"""
__version__ = 1.0
__name__ = 'GC_TLA'

from . import base_plopper
#from .base_plopper import (findReplaceRegex, Plopper, LibE_Plopper, ECP_Plopper, Polybench_Plopper,
#                           Dummy_Plopper, LazyPlopper, )
from . import base_problem
#from .base_problem import (setWhenDefined, BaseProblem, import_method_builder, libe_problem_builder,
#                           dummy_problem_builder, ecp_problem_builder, polybench_problem_builder, )
from . import gc_tla_utils
#from .gc_tla_utils import (load_from_problem, load_problem_module, load_without_problem, )
