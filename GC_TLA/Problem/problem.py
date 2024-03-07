import time
import pathlib
import warnings
import enum
# MODULE dependencies
import numpy as np
import pandas as pd
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real
from sdv.constraints import ScalarRange
# Own library
from GC_TLA.Utils.factory_configurable import FactoryConfigurable
from GC_TLA.Plopper.architecture import Architecture as Arch
from GC_TLA.Plopper.executor import Executor
from GC_TLA.Plopper.oracle_executor import OracleExecutor
from GC_TLA.Plopper.plopper import Plopper

class ProblemReturnMode(enum.Enum):
    ytopt = enum.auto()
    gptune = enum.auto()

    @classmethod
    def from_str(cls, string):
        l_string = string.lower()
        mode_keys = list(cls.__members__.keys())
        if l_string not in mode_keys:
            raise ValueError(f"Value '{string}' cannot be converted into ProblemReturnMode (options: {mode_keys})")
        return list(cls.__members__.values())[mode_keys.index(l_string)]

class Problem(FactoryConfigurable):
    def __init__(self, architecture, executor, plopper,
                 tunable_params, problem_identifier,
                 silent=False, returnmode='ytopt', logfile=None, logfile_clobber=False,
                 log_result_col='objective', log_time_col='elapsed_sec'):
        super().__init__()
        """
            Subclasses should additionally define:
                * output_space
                * constraints
        """
        # Assign these values after confirming they are instances of proper classes
        assert isinstance(architecture, Arch), "Architecture must be instance of GC_TLA Architecture"
        self.architecture = architecture
        assert isinstance(executor, Executor), "Executor must be an instance of GC_TLA Executor"
        self.executor = executor
        assert isinstance(plopper, Plopper), "Plopper must be an instance of GC_TLA Plopper"
        self.plopper = plopper

        assert isinstance(tunable_params, CS.ConfigurationSpace), "Tunable Parameters must be an instance of ConfigSpace ConfigurationSpace"
        # Derive tuning space size
        self.tuning_space_size = 1
        for param in tunable_params.get_hyperparameters():
            if isinstance(param, CSH.CategoricalHyperparameter):
                self.tuning_space_size *= len(param.choices)
            elif isinstance(param, CSH.OrdinalHyperparameter):
                self.tuning_space_size *= len(param.sequence)
            elif isinstance(param, CSH.Constant):
                continue # *= 1
            else:
                warnings.warn(f"Unknown parameter type {type(param)} for parameter {param}",UserWarning)
        self.tunable_params = tunable_params
        self.tunable_param_set = set(tunable_params) # __iter__() only goes over names, not *Hyperparameter objects
        self.n_params = len(self.tunable_param_set)

        assert problem_identifier is not None, "Problem Identifier must not be None"
        self.problem_identifier = problem_identifier
        # Derive name from class (possibly sub-classed) name and specific problem identifier
        self.name = f"{self.__class__.__name__}<{self.problem_identifier}>"

        self.silent = silent
        if type(returnmode) is str:
            returnmode = ProblemReturnMode.from_str(returnmode)
        elif not isinstance(returnmode, ProblemReturnMode):
            raise TypeError(f"returnmode must be ProblemReturnMode or string interpretable as ProblemReturnMode")
        self.returnmode = returnmode
        if logfile is None:
            self.logfile = None
        else:
            self.logfile = pathlib.Path(logfile)
            # Delete previous log with level of care indicated by arguments
            if not logfile_clobber and self.logfile.exists():
                raise ValueError(f"Will not clobber existing self-log file {self.logfile}")
            else:
                self.logfile.unlink(missing_ok=True)
            self.result_col = log_result_col
            self.time_col = log_time_col

        # Sentinel value to ensure timing starts upon first evaluation
        self.start_time = None

    def rename(self, base_name):
        self.name = f"{self.__class__.__name__}<{self.problem_identifier}>"

    def __str__(self):
        components = [self.name,
                      f"Space with {self.n_params} parameters forming {self.tuning_space_size} configurations",
                      str(self.architecture),
                      str(self.executor)]
        # Plopper str won't initially be indented all the way
        modified_plopper_string = "\n\t".join(str(self.plopper).split("\n"))
        components.append(modified_plopper_string)
        return "\n\t".join(components)

    def _log(self, config, result):
        if self.logfile is None:
            raise ValueError("No logfile configured")
        point[self.result_col] = result
        point[self.time_col] = time.time() - self.start_time
        frame = pd.DataFrame(data=[point], columns=list(point.keys()))
        if self.logfile.exists():
            frame = pd.read_csv(self.logfile).append(frame, ignore_index=True)
        frame.to_csv(self.logfile, index=False)

    def evaluateConfiguration(self, config, *args, **kwargs):
        configList = [config[param] for param in self.tunable_params]
        # Ensure any omitted values are injected into kwargs
        omitted_configuration = set(config.keys()).difference(self.tunable_param_set)
        kwargs['dropped_config_params'] = dict((k,config[k]) for k in omitted_configuration)

        if not self.silent:
            print(f"Evaluate Configuration: {config}")

        # Ensure starting time is set
        if self.start_time is None:
            self.start_time = time.time()

        # Use oracle evaluation only when proper class/subclass and use_oracle=True passed in
        if isinstance(self.executor, OracleExecutor) and \
            'use_oracle' in kwargs and \
            kwargs['use_oracle']==True:
            # CIRCUMVENT plopper to directly interact with OracleExecutor
            # Passes along as_rank=True, single_return=True, etc when present (along with other items a subclass may require)
            result = self.executor.oracleSearch(configList, *args, **kwargs)
        else:
            # Passes along use_raw_template and other items a subclass may require
            # But strip destination if present
            if 'destination' in kwargs.keys():
                destination = kwargs.pop('destination')
            else:
                destination = None
            result = self.plopper.templateExecute(destination, configList, *args, **kwargs)

        if not self.silent:
            print(f"Evaluation Result: {config} --> {result}")

        if self.logfile is not None:
            self._log(configList, result)

        if self.returnmode == ProblemReturnMode.ytopt:
            return result
        elif self.returnmode == ProblemReturnMode.gptune:
            return [result]
        else:
            raise ValueError(f"Return mode {self.returnmode} not implemented!")

class RuntimeProblem(Problem):
    """
        Common problem definition that includes an output space as defined below
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_space = ParamSpace([Real(0.0, inf, name='time')])

