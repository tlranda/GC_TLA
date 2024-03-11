import pathlib
# Dependent modules
from ConfigSpace import ConfigurationSpace as CS
from ConfigSpace.hyperparameters import (CategoricalHyperparameter as Categorical, OrdinalHyperparameter as Ordinal)
from sdv.constraints import ScalarRange
import numpy as np
# Own library
from GC_TLA.utils import (Factory, FindReplaceRegex)
from GC_TLA.plopper import (Arch, Executor, OracleExecutor, Plopper)
from GC_TLA.problem import RuntimeProblem

"""
    Structure

    The ultimate top-level factory that will be importable from this file is:
        Factory(RuntimeProblem)

    The RuntimeProblem requires extensive configuration, so the Factory is subclassed to facilitate proper building:
        Positional arguments for Arch, Executor, and Plopper need to be factory-built at build-time based on name
        Append tunable args as a positional parameter

        Architecture:
            Use default Arch() instance at build time, for now there is no need to update it

        Executor:
            Requires a subclass of OracleExecutor (details closer to implementation)
            Requires a Factory subclass to handle optional oracle initialization

        Plopper:
            Requires a subclass of Plopper (details closer to implementation)
            Requires a Factory subclass to statically set several values, including the polybench dataset flag based on instance's size
            RECEIVES arch/executor from RuntimeProblem's factory
"""

# Hyperparameters for Syr2k Search
tunable_params = CS()
tunable_params.add_hyperparameters([
    Categorical(name='P0',choices=["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "], default_value=" "),
    Categorical(name='P1',choices=["#pragma clang loop(i1) pack array(B) allocate(malloc)", " "], default_value=" "),
    Categorical(name='P2',choices=["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "], default_value=' '),
    Ordinal(name='P3',sequence=['4','8','16','20','32','50','64','80','96','100','128'], default_value='96'),
    Ordinal(name='P4',sequence=['4','8','16','20','32','50','64','80','100','128','2048'], default_value='2048'),
    Ordinal(name='P5',sequence=['4','8','16','20','32','50','64','80','100','128','256'], default_value='256'),
    ])
# Available sizes to be built
problem_mapping = {
    20: ("N", "MINI"),
    60: ("S", "SMALL"),
    130: ("SM", "SM"),
    200: ("M", "MEDIUM"),
    600: ("ML", "ML"),
    1000: ("L", "LARGE"),
    2000: ("XL", "EXTRALARGE"),
    3000: ("H", "HUGE"),
    }
# Name the module imports as -- formatted with the FIRST argument in the problem mapping tuple above (ie: syr2k_N)
IMPORT_AS = 'syr2k'
inv_mapping = dict((v[0],k) for (k,v) in problem_mapping.items())
inv_mapping.update(dict((v[1],k) for (k,v) in problem_mapping.items()))
import_to_dataset = dict((v[0],v[1]) for v in problem_mapping.values())
# SDV needs to know a constraint for possible sizes as well
constraints = [ScalarRange(column_name='input',
                           low_value=min(problem_mapping.keys()),
                           high_value=max(problem_mapping.keys()),
                           strict_boundaries=False),
              ]

# Subclass of Factory to change build() behavior
class Syr2kFactory(Factory):
    def build(self, name, *args, **kwargs):
        # Have to instantiate arch, executor and plopper at build time based on build() parameters
        new_args = list()
        if self.arch_factory is None:
            raise ValueError("Sub-Factory for arch was not configured!")
        new_args.append(self.arch_factory.build(name))
        if self.exe_factory is None:
            raise ValueError("Sub-Factory for exe was not configured!")
        new_args.append(self.exe_factory.build(name))
        if self.plopper_factory is None:
            raise ValueError("Sub-Factory for plopper was not configured!")
        new_args.append(self.plopper_factory.build(name,
                                                   architecture=new_args[0],
                                                   executor=new_args[1]))
        new_args.append(self.tunable_params)
        # Append mapping identifier through inverted dictionary
        new_args.append(self.inv_mapping[name.rsplit('_',1)[1]])
        return super().build(name, *new_args, **kwargs)
Syr2kFactory._configure(arch_factory=None, exe_factory=None, plopper_factory=None,
                        inv_mapping=inv_mapping, tunable_params=tunable_params)
# After this initialization, we just need to supply the three factories, then this factory is ready to produce instances
syr2k_instance_factory = Syr2kFactory(RuntimeProblem,
                                      factory_name=IMPORT_AS,
                                      initial_configure={'constraints':constraints,
                                                         'problem_mapping':problem_mapping,
                                                         })
# Architecture factory is a shallow pass, nothing special to do here
syr2k_arch_factory = Factory(Arch)
syr2k_instance_factory._update_from_core(arch_factory=syr2k_arch_factory)

# Executor factory works on a subclass of the OracleExecutor, as some oracle data is present for Syr2k
"""
    Must subclass to redefine getMetric()
    Runtime for Syr2k requires getMetric's aggregator_fn to exist, prior version was lambda x: np.mean(x[1:]) so we'll make that the default
    Factory must build executor as the proper Oracle when the oracle data exists
"""
class Syr2kExecutor(OracleExecutor):
    def getMetric(self, logfile, outfile, attempt, *args, aggregator_fn=None, **kwargs):
        if logfile is None:
            # Parsing failed, return None
            return None
        if aggregator_fn is None:
            # Default to same behavior as original GC_TLA experiments
            aggregator_fn = lambda x: np.mean(x[1:])
        with open(logfile, 'r') as f:
            # Take nonempty lines from input
            data = [_.rstrip() for _ in f.readlines() if len(_.rstrip()) > 0]
            try:
                # Data was 3 floating point values at the end of the file, delimited by newlines
                return aggregator_fn([float(x) for x in data[-3:]])
            except:
                # Parsing failed, return None
                return None
# Optionally provides an oracle file to support oracle evaluations -- but only for sizes where we have an oracle file
class Syr2kExecutorFactory(Factory):
    def build(self, name, *args, **kwargs):
        if name in self.oracles.keys():
            kwargs['oracle_path'] = self.oracles[name]
        return super().build(name, *args, **kwargs)
#Syr2kExecutorFactory._configure(oracles=dict())
# Set up the other oracle values here
syr2k_exe_factory = Syr2kExecutorFactory(Syr2kExecutor,
                                         initial_kwargs={'oracle_sort_keys': ['objective'],
                                                         'oracle_match_cols': list(tunable_params),
                                                         'oracle_return_cols': ['objective'],
                                                        },)
# Define oracles for the factory
oracles = {f'{IMPORT_AS}_SM': pathlib.Path(__file__).parents[1].joinpath('Data/polybench/syr2k/oracle/all_SM.csv'),
           f'{IMPORT_AS}_XL': pathlib.Path(__file__).parents[1].joinpath('Data/polybench/syr2k/oracle/all_XL.csv'),
          }
syr2k_exe_factory._update_from_core(oracles=oracles)
syr2k_instance_factory._update_from_core(exe_factory=syr2k_exe_factory)
"""
        Plopper:
            Dynamically set dataset based on instance's size
"""

class Syr2kPlopper(Plopper):
    def buildTemplateCmds(self, outfile, *args, lookup_match_substitution=None, **kwargs):
        return [f"echo clang {outfile} {pathlib.Path(__file__).parents[0].joinpath('polybench.c')} -I{pathlib.Path(__file__).parents[0]} {self.dataset} "
                "-DOPLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm -polly-process-unprofitable "
                f"-mllvm -polly-use-llvm-names -ffast-math -march=native -o {outfile.with_suffix('')}",]

    def buildExecutorCmds(self, outfile, *args, **kwargs):
        #return [f"srun -n1 {outfile.with_suffix('')}",]
        # For debugging purposes, circumvent actually evaluating and just force values into the log file
        return [f"echo srun -n1 {outfile.with_suffix('')}",
                "echo -20943", # Ignored by aggregator_fn
                "echo 20.00", # Together the average should be 20.24
                "echo 20.48",]
# The dataset attribute is not set, it has to be manufactured by a factory
class Syr2kPlopperFactory(Factory):
    def build(self, name, *args, **kwargs):
        instance = super().build(name, *args, **kwargs)
        dataset_name = self.import_to_dataset[name.rsplit("_",1)[1]]
        instance.dataset = f"-D{dataset_name}_DATASET"
        return instance

syr2k_FindReplaceRegex = FindReplaceRegex(r"(P[0-9]+)", prefix=("#","",))

Syr2kPlopperFactory._configure(import_to_dataset=import_to_dataset)
syr2k_plopper_factory = Syr2kPlopperFactory(Syr2kPlopper,
                                            initial_args=[pathlib.Path('mmp.c')],
                                            initial_kwargs={'output_extension': '.c',
                                                            'findReplace': syr2k_FindReplaceRegex,})
syr2k_instance_factory._update_from_core(plopper_factory=syr2k_plopper_factory)

# Finally, we can update the import method for this file to utilize the instance factory
__getattr__ = syr2k_instance_factory.getattr

