import pathlib
import copy
from collections.abc import Mapping
from itertools import product as itertools_product
from math import ceil as math_ceil
# Dependent modules
from ConfigSpace import ConfigurationSpace as CS
from ConfigSpace.hyperparameters import (CategoricalHyperparameter as Categorical, Constant, OrdinalHyperparameter as Ordinal)
from sdv.constraints import ScalarRange
import numpy as np
# Own library
from GC_TLA.utils import Factory
from GC_TLA.plopper import (Arch, OracleExecutor, EphemeralPlopper)
from GC_TLA.problem import RuntimeProblem

# Hyperparameters are fixed
tunable_params = CS()
tunable_params.add_hyperparameters([
    Categorical(name="VF", choices=[2**_ for _ in range(7)], default_value=1),
    Categorical(name="IF", choices=[2**_ for _ in range(5)], default_value=1)
])

IMPORT_AS='neurovectorizer'

class neurovecInstanceFactory(Factory):
    def build(self, name, *args, **kwargs):
        # For now, identifier is whatever you request it to be
        identifier = name.split("_",1)[1]
        new_args = []
        if 'architecture' in kwargs.keys():
            new_args.append(kwargs['architecture'])
            # Prevent propagation issue -- constructed object doesn't need a second reference to this attribute
            del kwargs['architecture']
        else:
            if self.arch_factory is None:
                raise ValueError("Sub-factory for architecture was not configured!")
            new_args.append(self.arch_factory.build(name))
        if self.exe_factory is None:
            raise ValueError("Sub-factory for executor was not configured!")
        new_args.append(self.exe_factory.build(name))
        if self.plopper_factory is None:
            raise ValueError("Sub-factory for plopper was not configured!")
        new_args.append(self.plopper_factory.build(name,
                            architecture=new_args[0],
                            executor=new_args[1]))
        new_args.append(self.tunable_params)
        new_args.append(identifier)
        instance = super().build(name, *new_args, **kwargs)
        instance.silent = True
        return instance
neurovecInstanceFactory._configure(arch_factory=None,
                                   exe_factory=None,
                                   plopper_factory=None,
                                   tunable_params=tunable_params)
neurovec_instance_factory = neurovecInstanceFactory(RuntimeProblem,
                                   factory_name=IMPORT_AS,
                                   #debug_class_construction=True,
                                   )
# Standard architecture should be OK
neurovec_arch_factory = Factory(Arch)
neurovec_instance_factory._update_from_core(arch_factory=neurovec_arch_factory)

# Executor must be an OracleExecutor as I cannot execute these problems and only rely on previous observations
class neurovecExecutorFactory(Factory):
    def build(self, name, *args, **kwargs):
        if name in self.oracles.keys():
            kwargs['oracle_path'] = self.oracles[name]
        return super().build(name, *args, **kwargs)
neurovec_exe_factory = neurovecExecutorFactory(OracleExecutor,
                                               initial_kwargs={'oracle_sort_keys': ['runtime'],
                                                               'oracle_match_cols': list(tunable_params),
                                                               'oracle_return_cols': ['runtime'],},)
# Define oracles for the factory
oracles = {f'{IMPORT_AS}_titan': pathlib.Path(__file__).parents[0].joinpath('neurovec_runtimes_titan.csv')}
neurovec_exe_factory._update_from_core(oracles=oracles)
neurovec_instance_factory._update_from_core(exe_factory=neurovec_exe_factory)

# Plopper is probably pretty straightfoward, standard OK?
neurovec_plopper_factory = Factory(EphemeralPlopper)
neurovec_instance_factory._update_from_core(plopper_factory=neurovec_plopper_factory)

# Finally, set import method for this module using the instance factory
__getattr__ = neurovec_instance_factory.getattr

