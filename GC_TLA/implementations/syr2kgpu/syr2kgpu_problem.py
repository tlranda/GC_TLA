import pathlib
# Dependent modules
from ConfigSpace import ConfigurationSpace as CS
from ConfigSpace.hyperparameters import (CategoricalHyperparameter as Categorical, OrdinalHyperparameter as Ordinal)
from sdv.constraints import ScalarRange
import numpy as np
# Own library
from GC_TLA.utils import (Factory, FindReplaceRegex)
from GC_TLA.plopper import (Arch, Executor, MetricIDs, OracleExecutor, Plopper)
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

# Hyperparameters for Syr2kGPU Search
tunable_params = CS()
tunable_params.add_hyperparameters([
    #Categorical(name='P0',choices=["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "], default_value=" "),
    #Categorical(name='P1',choices=["#pragma clang loop(i1) pack array(B) allocate(malloc)", " "], default_value=" "),
    #Categorical(name='P2',choices=["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "], default_value=' '),
    # Above do not make sense on GPU and will decrease performance if carelessly replicated, exclude from space

    Ordinal(name='P3',sequence=['4','8','16','20','32','50','64','80','96','100','128'], default_value='96'),
    Ordinal(name='P4',sequence=['4','8','16','20','32','50','64','80','100','128','2048'], default_value='2048'),
    # Pick 80,128: |P4| = 2, max P4 = 8    (4)
    # Pick 50,64: |P4| = 3, max P4 = 16    (6)
    # Pick 20,32: |P4| = 5, max P4 = 32    (10)
    # Pick 16: |P4| = 7, max P4 = 64       (7)
    # Pick 8: |P4| = 10, max P4 = 128      (10)
    # 4 does not enable any further values (10)
    # Total expected valid P3+P4 combinations: 47 (naive expectation: 11x11 = 121, 39% coverage)
    Ordinal(name='P5',sequence=['4','8','16','20','32','50','64','80','100','128','256'], default_value='256'),
    # Actual search space size is 47x11 = 517, not 11x11x11=1331
    ])
# Available sizes to be built
problem_mapping = {
    256: ("N", "MINI"),
    512: ("S", "SMALL"),
    1024: ("SM", "SM"),
    2048: ("M", "MEDIUM"),
    4096: ("ML", "ML"),
    8192: ("L", "LARGE"),
    16384: ("XL", "EXTRALARGE"),
    32768: ("H", "HUGE"),
    }
# Name the module imports as -- formatted with the FIRST argument in the problem mapping tuple above (ie: syr2k_gpu_N)
IMPORT_AS = 'syr2kgpu'
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
class Syr2kGPUFactory(Factory):
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
Syr2kGPUFactory._configure(arch_factory=None, exe_factory=None, plopper_factory=None,
                           inv_mapping=inv_mapping, tunable_params=tunable_params)
# After this initialization, we just need to supply the three factories, then this factory is ready to produce instances
syr2kgpu_instance_factory = Syr2kGPUFactory(RuntimeProblem,
                                            factory_name=IMPORT_AS,
                                            initial_configure={'constraints':constraints,
                                                               'problem_mapping':problem_mapping,
                                                              })
# Architecture factory is a shallow pass, nothing special to do here
syr2kgpu_arch_factory = Factory(Arch)
syr2kgpu_instance_factory._update_from_core(arch_factory=syr2kgpu_arch_factory)

# Executor factory works on a subclass of the OracleExecutor, as some oracle data is present for Syr2k
"""
    Must subclass to redefine getMetric()
    Runtime for Syr2k requires getMetric's aggregator_fn to exist, prior version was lambda x: np.mean(x[1:]) so we'll make that the default
    Factory must build executor as the proper Oracle when the oracle data exists

    GPU kernel can also fail on some invalid configurations, so we check/report that as well
"""
class Syr2kGPUExecutor(OracleExecutor):
    def getMetric(self, logfile, outfile, attempt, *args, aggregator_fn=None, **kwargs):
        if logfile is None:
            # Parsing failed, return None
            return None
        if aggregator_fn is None:
            # Default to same behavior as original GC_TLA experiments
            aggregator_fn = lambda x: np.mean(x[1:]) if len(x) > 2 else np.mean(x)
        with open(logfile, 'r') as f:
            # Take nonempty lines from input
            data = [_.rstrip() for _ in f.readlines() if len(_.rstrip()) > 0]
            if any([_.startswith('ERRORS') for _ in data]):
                # Execution failed due to inaccuracy
                return self.infinity[MetricIDs.NotOK]
            try:
                # Data was 3 floating point values at the end of the file, delimited by newlines
                # TODO: Should check for actually solving correctly
                return aggregator_fn([float(x) for x in data])
            except:
                # Parsing failed, return None
                return None
# Optionally provides an oracle file to support oracle evaluations -- but only for sizes where we have an oracle file
class Syr2kGPUExecutorFactory(Factory):
    def build(self, name, *args, **kwargs):
        if name in self.oracles.keys():
            kwargs['oracle_path'] = self.oracles[name]
        return super().build(name, *args, **kwargs)
#Syr2kGPUExecutorFactory._configure(oracles=dict())
# Set up the other oracle values here
syr2kgpu_exe_factory = Syr2kGPUExecutorFactory(Syr2kGPUExecutor,
                                               initial_kwargs={'oracle_sort_keys': ['objective'],
                                                               'oracle_match_cols': list(tunable_params),
                                                               'oracle_return_cols': ['objective'],
                                                               'evaluation_tries': 3,
                                                              },)
# Define oracles for the factory
oracles = {f'{IMPORT_AS}_SM': pathlib.Path(__file__).parents[3].joinpath(f'Data/polybench/{IMPORT_AS}/oracle/all_SM.csv'),
           f'{IMPORT_AS}_XL': pathlib.Path(__file__).parents[3].joinpath(f'Data/polybench/{IMPORT_AS}/oracle/all_XL.csv'),
          }
syr2kgpu_exe_factory._update_from_core(oracles=oracles)
syr2kgpu_instance_factory._update_from_core(exe_factory=syr2kgpu_exe_factory)
"""
    Plopper:
        Dynamically set dataset based on instance's size
"""

class Syr2kGPUPlopper(Plopper):
    def buildTemplateCmds(self, outfile, *args, lookup_match_substitution=None, **kwargs):
        return [f"nvcc {outfile} -I{pathlib.Path(__file__).parents[0]} {self.dataset} "
                f"-DPOLYBENCH_TIME -O3 -o {outfile.with_suffix('')}",]

    def buildExecutorCmds(self, outfile, *args, **kwargs):
        return [f"{outfile.with_suffix('')}",]
        # For debugging purposes, circumvent actually evaluating and just force values into the log file
        #return [f"echo srun -n1 {outfile.with_suffix('')}",
        #        "echo -20943", # Ignored by aggregator_fn
        #        "echo 20.00", # Together the average should be 20.24
        #        "echo 20.48",]
# The dataset attribute is not set, it has to be manufactured by a factory
class Syr2kGPUPlopperFactory(Factory):
    def build(self, name, *args, **kwargs):
        instance = super().build(name, *args, **kwargs)
        dataset_name = self.import_to_dataset[name.rsplit("_",1)[1]]
        instance.dataset = f"-D{dataset_name}_DATASET"
        return instance

syr2kgpu_FindReplaceRegex = FindReplaceRegex(r"(P[0-9]+)", prefix=("#","",))

Syr2kGPUPlopperFactory._configure(import_to_dataset=import_to_dataset)
syr2kgpu_plopper_factory = Syr2kGPUPlopperFactory(Syr2kGPUPlopper,
                                                  initial_args=[pathlib.Path(__file__).parents[0].joinpath('mmp.cu')],
                                                  initial_kwargs={'output_extension': '.cu',
                                                                  'findReplace': syr2kgpu_FindReplaceRegex,})
syr2kgpu_instance_factory._update_from_core(plopper_factory=syr2kgpu_plopper_factory)

# Finally, we can update the import method for this file to utilize the instance factory
__getattr__ = syr2kgpu_instance_factory.getattr

