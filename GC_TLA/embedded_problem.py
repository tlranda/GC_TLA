import pathlib
from collections.abc import Mapping
# Dependent modules
from ConfigSpace import ConfigurationSpace as CS
from ConfigSpace.hyperparameters import (CategoricalHyperparameter as Categorical, OrdinalHyperparameter as Ordinal, UniformFloatHyperparameter as UniformFloat)
from sdv.constraints import ScalarRange
import numpy as np
# Own library
from GC_TLA.utils import (Factory, FindReplaceRegex)
from GC_TLA.plopper import (Arch, Executor, OracleExecutor, EphemeralPlopper)
from GC_TLA.problem import RuntimeProblem

def build_configuration_space_based_on_embedding_size(n, seed=None):
    tunable_params = CS(seed=seed)
    parameters = [Categorical(name='P0', choices=["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "], default_value=" "),
                  Categorical(name='P1',choices=["#pragma clang loop(i1) pack array(B) allocate(malloc)", " "], default_value=" "),
                  Categorical(name='P2',choices=["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "], default_value=' '),
                  Ordinal(name='P3',sequence=['4','8','16','20','32','50','64','80','96','100','128'], default_value='96'),
                  Ordinal(name='P4',sequence=['4','8','16','20','32','50','64','80','100','128','2048'], default_value='2048'),
                  Ordinal(name='P5',sequence=['4','8','16','20','32','50','64','80','100','128','256'], default_value='256'),
    ]
    parameters.extend([UniformFloat(name=f'emb_dim_{i}', lower=-5., upper=5.) for i in range(n)])
    tunable_params.add_hyperparameters(parameters)
    return tunable_params

# Problem identifiers are a Syr2k size + a number of embedding dimensions
class EmbeddedProblemIDMapper(Mapping):
    APP_SCALES = [20, 60, 130, 200, 600, 1000, 2000, 3000]
    APP_SCALE_NAMES = [('N','MINI'),
                       ('S','SMALL'),
                       ('SM','SM'),
                       ('M','MEDIUM'),
                       ('ML','ML'),
                       ('L','LARGE'),
                       ('XL','EXTRALARGE'),
                       ('H','HUGE')]

    @property
    def app_scale_range(self):
        return min(self.APP_SCALES), max(self.APP_SCALES)

    def __getitem__(self, key):
        inverted = []
        if type(key) is str:
            converted = []
            for field in key.split('_'):
                try:
                    converted.append(int(field))
                except ValueError:
                    converted.append(field)
            fields = tuple(converted)
        elif type(key) is tuple:
            fields = key
        else:
            raise ValueError('Keys should be tuples or strings')
        if len(fields) != 2:
            raise KeyError('Did not identify 2 fields (app scale, # embeddings')
        if fields[0] in self.APP_SCALES:
            inverted.append(self.APP_SCALE_NAMES[self.APP_SCALES.index(fields[0])])
        else:
            found = False
            for idx, pair in enumerate(self.APP_SCALE_NAMES):
                if fields[0] in pair:
                    inverted.append(self.APP_SCALES[idx])
                    found = True
                    break
            if not found:
                raise KeyError(f"App Scale ({fields[0]}) not in known app scales (as integer: {self.APP_SCALES}) or (as string: {self.APP_SCALE_NAMES})")
        inverted.append(fields[1])
        return tuple(inverted)

    def __iter__(self):
        return [(scale, 1) for scale in self.APP_SCALES]

    def __len__(self):
        return len(self.APP_SCALES)

embeddedProblemID_mapping = EmbeddedProblemIDMapper()
min_size, max_size = embeddedProblemID_mapping.app_scale_range
constraints = [ScalarRange(column_name='input',
                           low_value=min_size,
                           high_value=max_size,
                           strict_boundaries=False)]
IMPORT_AS='embedded'

class EmbeddedInstanceFactory(Factory):
    def build(self, name, *args, **kwargs):
        # Identify from name with mapping
        name_size = name.split("_",1)[1]
        identifier = self.mapping[name_size]
        size, n_elems = identifier
        if type(size) is str:
            size, n_elems = self.mapping[identifier]
        # Update tunable params
        tunable_params = build_configuration_space_based_on_embedding_size(n_elems)
        self._update_from_core(tunable_params=tunable_params)
        new_args = list()
        if self.arch_factory is None:
            raise ValueError("Sub-factory for arch was not configured!")
        new_args.append(self.arch_factory.build(name))
        if self.exe_factory is None:
            raise ValueError("Sub-factory for exe was not configured!")
        new_args.append(self.exe_factory.build(name))
        if self.plopper_factory is None:
            raise ValueError("Sub-factory for plopper was not configured!")
        new_args.append(self.plopper_factory.build(name,
                                                   architecture=new_args[0],
                                                   executor=new_args[1],))
        new_args.append(tunable_params)
        new_args.append(identifier)
        return super().build(name, *new_args, **kwargs)

EmbeddedInstanceFactory._configure(mapping=embeddedProblemID_mapping)
embedded_instance_factory = EmbeddedInstanceFactory(RuntimeProblem,
                                                    factory_name=IMPORT_AS,
                                                    initial_configure={'constraints': constraints,
                                                    'problem_mapping': embeddedProblemID_mapping,
                                                    },)

# Shallow pass, nothing based on architecture
embedded_arch_factory = Factory(Arch)
embedded_instance_factory._update_from_core(arch_factory=embedded_arch_factory)

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
        name_with_scale = "_".join(name.split("_",2)[:2])
        if name_with_scale in self.oracles.keys():
            kwargs['oracle_path'] = self.oracles[name_with_scale]
        return super().build(name, *args, **kwargs)
#Syr2kExecutorFactory._configure(oracles=dict())
# Set up the other oracle values here
syr2k_exe_factory = Syr2kExecutorFactory(Syr2kExecutor,
                                         initial_kwargs={'oracle_sort_keys': ['objective'],
                                                         'oracle_match_cols': [f'P{_}' for _ in range(6)],
                                                         'oracle_return_cols': ['objective'],
                                                        },)
# Define oracles for the factory
oracles = {f'{IMPORT_AS}_SM': pathlib.Path(__file__).parents[1].joinpath('Data/polybench/syr2k/oracle/all_SM.csv'),
           f'{IMPORT_AS}_XL': pathlib.Path(__file__).parents[1].joinpath('Data/polybench/syr2k/oracle/all_XL.csv'),
          }
syr2k_exe_factory._update_from_core(oracles=oracles)
embedded_instance_factory._update_from_core(exe_factory=syr2k_exe_factory)

# Plopper should not actually run
embedded_plopper_factory = Factory(EphemeralPlopper)
embedded_instance_factory._update_from_core(plopper_factory=embedded_plopper_factory)

__getattr__ = embedded_instance_factory.getattr
