import numpy as np, pandas as pd
from autotune import TuningProblem # Merge destination for BaseProblem
from autotune.space import *
from skopt.space import Real, Integer, Categorical
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
# NEW IMPORTS
from ConfigSpace import EqualsCondition
# UPDATE SDV
from sdv.constraints import ScalarRange
import inspect
import time
import os
# NEW PLOPPER
from GC_TLA.base_plopper import ECP_Plopper, Polybench_Plopper, LibE_Plopper, Dummy_Plopper

parameter_lookups = {'UniformInt': CSH.UniformIntegerHyperparameter,
                     'NormalInt': CSH.NormalIntegerHyperparameter,
                     'UniformFloat': CSH.UniformFloatHyperparameter,
                     'NormalFloat': CSH.NormalFloatHyperparameter,
                     'Ordinal': CSH.OrdinalHyperparameter,
                     'Categorical': CSH.CategoricalHyperparameter,
                     'Constant': CSH.Constant
                    }

class NoneStandIn():
    pass

class setWhenDefined():
    """
        Subclassing this class allows you to define class attributes that MATCH the names
        of local attributes (including ones OUTSIDE the function signature, so be careful
        about that)
        NOTE: You must EXPLICITLY reserve args and kwargs local values to be *args, **kwargs,
        or else this may not behave as expected
    """
    def overrideSelfAttrs(self):
        SWD_ignore = set(['self', 'args', 'kwargs', 'SWD_ignore']) # We're going to omit these frequently
        frame = inspect.currentframe().f_back
        flocals = frame.f_locals # Parent stack function local variables
        fcode = frame.f_code # Code object for parent stack function
        # LOCALS
        values = dict((k,v) for (k,v) in flocals.items() if k not in SWD_ignore)
        # VARARGS
        if 'args' in flocals.keys() and len(flocals['args']) > 0:
            values.update({'varargs': flocals['args']})
        # KWARGS
        if 'kwargs' in flocals.keys():
            values.update(dict((k,v) for (k,v) in flocals['kwargs'].items() if k not in SWD_ignore))
        # Get names of all arguments from your __init__ method and subtract the few we know to ignore
        specified_values = fcode.co_varnames[:fcode.co_argcount]
        override = set(specified_values).difference(SWD_ignore)
        for attrname in override:
            # When the current value is None but we have a class default, choose that default
            if values[attrname] is None and hasattr(self, attrname):
                values[attrname] = getattr(self, attrname)
        # Apply values to the attributes of this instance
        for k,v in values.items():
            setattr(self, k, v)

# Should be merge-able with Autotune's TuningProblem
class BaseProblem(setWhenDefined):
    # Many subclasses will override the pre-init space with default attributes
    def __init__(self, input_space: Space = None, space_alteration_callback: callable = None,
                 parameter_space: Space = None, output_space: Space = None,
                 problem_params: dict = None, problem_class: int = None,
                 plopper: object = None, constraints = None, models = None, name = None,
                 constants = None, silent = False, use_capital_params = False,
                 returnmode = 'ytopt', selflog = None, ignore_runtime_failure = False,
                 oracle = None, use_oracle = False, **kwargs):
        # Load problem attribute defaults when available and otherwise required (and None)
        self.overrideSelfAttrs()
        if self.space_alteration_callback is not None:
            self.space_alteration_callback()
        if self.name is None:
            self.name = self.__class__.__name__+'_size_'+str(problem_class)
        self.request_output_prefix = f"results_{problem_class}"
        # Find input space size
        prod = 1
        for param in self.input_space.get_hyperparameters():
            if type(param) == CS.CategoricalHyperparameter:
                prod *= len(param.choices)
            elif type(param) == CS.OrdinalHyperparameter:
                prod *= len(param.sequence)
            elif type(param) == CS.Constant:
                continue
            elif type(param) == CS.UniformIntegerHyperparameter:
                prod *= param.upper - param.lower
            else:
                # Could warn here, but it'll generate way too much output
                # This catches when we don't know how to get a # of configurations
                # As Normal range is not necessarily defined with strict ranges and floats are floats
                continue
        self.input_space_size = prod
        # Attributes
        # Add default known things to the params list for usage as field dict
        added_keys = ('input', 'runtime')
        if 'input' not in self.problem_params.keys():
            self.problem_params['input'] = 'float'
        if 'runtime' not in self.problem_params.keys():
            self.problem_params['runtime'] = 'float'
        self.params = list([k for k in self.problem_params.keys() if k not in added_keys])
        self.CAPITAL_PARAMS = [_.capitalize() for _ in self.params]
        self.n_params = len(self.params)
        if oracle is not None:
            self.initialize_oracle()
        if self.selflog is not None and os.path.exists(self.selflog):
            os.remove(self.selflog)
        self.time_start = time.time()

    def __str__(self):
        s = [self.name, f"Space has {self.input_space_size} configurations: {self.input_space}",
             f"Plopper: {self.plopper}", ]
        return "\n".join(s)

    def initialize_oracle(self):
        if type(self.oracle) is str:
            self.oracle = pd.read_csv(self.oracle)
        self.oracle = self.oracle.sort_values(by='objective')

    def oracle_search(self, parameterization):
        search = tuple(parameterization)
        n_matching_columns = (self.oracle[self.params].astype(str) == search).sum(1)
        full_match_idx = np.where(n_matching_columns == self.n_params)[0]
        if len(full_match_idx) == 0:
            raise ValueError(f"Failed to find tuple {list(search_equals)} in oracle data")
        objective = self.oracle.iloc[full_match_idx]['objective'].values[0]
        #print(f"All file rank: {full_match_idx[0]} / {len(self.oracle)}")
        return objective


    def seed(self, SEED):
        if self.input_space is not None:
            try:
                self.input_space.seed(SEED)
            except AttributeError:
                pass
        if self.parameter_space is not None:
            try:
                self.parameter_space.seed(SEED)
            except AttributeError:
                pass
        if self.output_space is not None:
            try:
                self.output_space.seed(SEED)
            except AttributeError:
                pass
        if self.plopper is not None:
            try:
                self.plopper.seed(SEED)
            except AttributeError:
                pass

    def condense_results(self, results):
        if len(results) > 1:
            return float(np.mean(results[1:]))
        else:
            return float(results[0])

    def objective(self, point: dict, *args, **kwargs):
        if point != {}:
            x = np.asarray_chkfinite([point[k] for k in self.params]) # ValueError if any NaN or Inf
        else:
            x = [] # Prevent KeyErrors when there are no points to parameterize
        # Some users may pass in additional keys via point. Ensure they are kept alive
        inkeys = set(point.keys())
        paramkeys = set(self.params)
        extrakeys = inkeys.difference(paramkeys)
        if len(extrakeys) > 0:
            kwargs['extrakeys'] = dict((k,point[k]) for k in extrakeys)
        if not self.silent:
            print(f"CONFIG: {point}")
        if self.use_oracle and self.oracle is not None:
            result = self.oracle_search(x)
        elif self.use_capital_params is not None and self.use_capital_params:
            result = self.plopper.findRuntime(x, self.CAPITAL_PARAMS, *args, **kwargs)
        else:
            result = self.plopper.findRuntime(x, self.params, *args, **kwargs)
        time_stop = time.time()
        if hasattr(result, '__iter__'):
            final = self.condense_results(result)
        else:
            final = result
        if not self.silent:
            if final == result:
                print(f"OUTPUT: {final}")
            else:
                print(f"OUTPUT: {result} --> {final}")
        if self.selflog is not None:
            point['objective'] = final
            point['elapsed_sec'] = time_stop - self.time_start
            frame = pd.DataFrame(data=[point], columns=list(point.keys()))
            if os.path.exists(self.selflog):
                logs = pd.read_csv(self.selflog)
                logs = logs.append(frame, ignore_index=True)
            else:
                logs = frame
            logs.to_csv(self.selflog, index=False)
        if self.returnmode == 'GPTune':
            return [final]
        elif self.returnmode == 'ytopt':
            return final
        else:
            raise ValueError(f"Unknown objective return mode {self.returnmode}!")

    @staticmethod
    def configure_space(parameterization, seed=None):
        # create an object of ConfigSpace
        space = CS.ConfigurationSpace(seed=seed)
        params_list = []
        for (p_type,p_kwargs) in parameterization:
            params_list.append(parameter_lookups[p_type](**p_kwargs))
        space.add_hyperparameters(params_list)
        return space

    def set_space(self, new_space):
        self.input_space = new_space
        prod = 1
        for param in self.input_space.get_hyperparameters():
            if type(param) == CS.CategoricalHyperparameter:
                prod *= len(param.choices)
            elif type(param) == CS.OrdinalHyperparameter:
                prod *= len(param.sequence)
            elif type(param) == CS.Constant:
                continue
            elif type(param) == CS.UniformIntegerHyperparameter:
                prod *= param.upper - param.lower
            else:
                # Could warn here, but it'll generate way too much output
                # This catches when we don't know how to get a # of configurations
                # As Normal range is not necessarily defined with strict ranges and floats are floats
                continue
        self.input_space_size = prod

def import_method_builder(clsref, lookup, default, oracles):
    def getattr_fn(name, default=default, oracles=oracles):
        # Prevent some bugs where things that normal python getattr SHOULD be called
        if name.startswith("__"):
            return
        if name == "input_space" or name == "space":
            return clsref.input_space
        prefixes = ["_", "class"]
        suffixes = ["Problem"]
        for pre in prefixes:
            if name.startswith(pre):
                name = name[len(pre):]
        for suf in suffixes:
            if name.endswith(suf):
                name = name[:-len(suf)]
        if name.lower().startswith("oracle"):
            name = name[6:].lstrip('_')
            if name not in oracles.keys():
                raise ValueError(f"Module defining '{clsref.__name__}' has no oracle for '{name}'")
            oracle = oracles[name]
            class_size = lookup[name]
            return clsref(class_size, oracle=oracle, use_oracle=True)
        elif name in lookup.keys():
            class_size = lookup[name]
            return clsref(class_size)
        elif name == "":
            return clsref(default)
        else:
            raise ValueError(f"Module defining '{clsref.__name__}' has no attribute '{name}'")
    return getattr_fn

# NEW PROBLEM BUILDER
def libe_problem_builder(lookup, input_space_definition, there, default=None, name="LibE_Problem", plopper_class=LibE_Plopper, oracles=dict(), **original_kwargs):
    class LibE_Problem(BaseProblem):
        input_space = input_space_definition
        parameter_space = None
        # THIS MAY CHANGE
        output_space = Space([Real(0.0, inf, name='time')])
        constraints = [ScalarRange(column_name='input',
                                   low_value=min([_[1] for _ in lookup.keys()]),
                                   high_value=max([_[1] for _ in lookup.keys()]),
                                   strict_boundaries=False)]
        dataset_lookup = lookup
        def __init__(self, class_size, **kwargs):
            # Allow  anything to be overridden by passing it in as top priority
            for k, v in original_kwargs.items():
                kwargs.setdefault(k,v)
            expect_kwargs = {'use_capital_params': True,
                             'problem_class': class_size,
                             'plopper': plopper_class(there+"/speed3d.sh", there, output_extension='.sh',
                                                      force_plot=True, nodes=class_size[0])
                            }
            for k, v in expect_kwargs.items():
                kwargs.setdefault(k,v)
            SpaceNotCustomized = "Problem space not customized, but customize_space() available. Set architecture information and call again."
            if hasattr(self, 'customize_space'):
                self.plopper = kwargs['plopper']
                try:
                    self.customize_space(class_size)
                except ValueError:
                    warnings.warn(SpaceNotCustomized)
            elif 'customize_space' in kwargs:
                self.plopper = kwargs['plopper']
                try:
                    kwargs['customize_space'](self, class_size)
                except ValueError:
                    warnings.warn(SpaceNotCustomized)
            if type(self.input_space) is not CS.ConfigurationSpace:
                self.input_space = BaseProblem.configure_space(self.input_space)
            self.problem_params = dict((p.lower(), 'categorical') for p in self.input_space.get_hyperparameter_names())
            self.categorical_cast = dict((p.lower(), 'str') for p in self.input_space.get_hyperparameter_names())
            super().__init__(**kwargs)
    LibE_Problem.__name__ = name
    inv_lookup = dict((v, k) for (k,v) in lookup.items())
    if default is None:
        default = inv_lookup['S_2']
    return import_method_builder(LibE_Problem, inv_lookup, default, oracles)

def dummy_problem_builder(lookup, input_space_definition, there, default=None, name="Dummy_Problem", plopper_class=Dummy_Plopper, oracles=dict(), **original_kwargs):
    if type(input_space_definition) is not CS.ConfigurationSpace:
        input_space_definition = BaseProblem.configure_space(input_space_definition)
    class Dummy_Problem(BaseProblem):
        input_space = input_space_definition
        parameter_space = None
        output_space = Space([Real(0.0, inf, name='time')])
        problem_params = dict((p.lower(), 'categorical') for p in input_space_definition.get_hyperparameter_names())
        categorical_cast = dict((p.lower(), 'str') for p in input_space_definition.get_hyperparameter_names())
        constraints = [ScalarRange(column_name='input', low_value=min(lookup.keys()), high_value=max(lookup.keys()), strict_boundaries=False)]
        dataset_lookup = lookup
        def __init__(self, class_size, **kwargs):
            # Allow anything to be overridden by passing it in as top priority
            for k, v in original_kwargs.items():
                kwargs.setdefault(k,v)
            expect_kwargs = {'use_capital_params': True,
                             'problem_class': class_size,
                             'dataset': class_size,
                             'plopper': plopper_class(),
                             'silent': False,
                            }
            for k, v in expect_kwargs.items():
                kwargs.setdefault(k,v)
            super().__init__(**kwargs)
        def objective(self, point, *args, **kwargs):
            return super().objective(point, self.dataset, *args, **kwargs)
        def O3(self):
            return super().objective({}, self.dataset, O3=True)
    Dummy_Problem.__name__ = name
    inv_lookup = dict((v[0], k) for (k,v) in lookup.items())
    if default is None:
        default = inv_lookup['S']
    return import_method_builder(Dummy_Problem, inv_lookup, default, oracles)

def ecp_problem_builder(lookup, input_space_definition, there, default=None, name="ECP_Problem", plopper_class=ECP_Plopper, oracles=dict(), **original_kwargs):
    if type(input_space_definition) is not CS.ConfigurationSpace:
        input_space_definition = BaseProblem.configure_space(input_space_definition)
    class ECP_Problem(BaseProblem):
        input_space = input_space_definition
        parameter_space = None
        output_space = Space([Real(0.0, inf, name='time')])
        problem_params = dict((p.lower(), 'categorical') for p in input_space_definition.get_hyperparameter_names())
        categorical_cast = dict((p.lower(), 'str') for p in input_space_definition.get_hyperparameter_names())
        constraints = [ScalarRange(column_name='input', low_value=min(lookup.keys()), high_value=max(lookup.keys()), strict_boundaries=False)]
        dataset_lookup = lookup
        def __init__(self, class_size, sourcefile='mmp.c', **kwargs):
            # Allow anything to be overridden by passing it in as top priority
            for k, v in original_kwargs.items():
                kwargs.setdefault(k,v)
            expect_kwargs = {'use_capital_params': True,
                             'problem_class': class_size,
                             'dataset': class_size,
                             'sourcefile': there+"/mmp.c",
                             'ignore_runtime_failure': False,
                            }
            if 'sourcefile' in kwargs:
                expect_kwargs['sourcefile'] = kwargs['sourcefile']
            if 'ignore_runtime_failure' in kwargs:
                expect_kwargs['ignore_runtime_failure'] = kwargs['ignore_runtime_failure']
            expect_kwargs['plopper'] = plopper_class(expect_kwargs['sourcefile'], there, output_extension=".c",
                                                     ignore_runtime_failure=expect_kwargs['ignore_runtime_failure'])
            for k, v in expect_kwargs.items():
                kwargs.setdefault(k,v)
            super().__init__(**kwargs)
        def objective(self, point, *args, **kwargs):
            return super().objective(point, self.dataset, *args, **kwargs)
        def O3(self):
            # Temporarily swap references
            old_source = self.plopper.sourcefile
            self.plopper.sourcefile = self.name.split('_',1)[0].lower()+".c"
            rvalue = super().objective({}, self.dataset, O3=True)
            self.plopper.sourcefile = old_source
            return rvalue
    ECP_Problem.__name__ = name
    inv_lookup = dict((v[0], k) for (k,v) in lookup.items())
    if default is None:
        default = inv_lookup['S']
    return import_method_builder(ECP_Problem, inv_lookup, default, oracles)

def polybench_problem_builder(lookup, input_space_definition, there, default=None, name="Polybench_Problem", plopper_class=Polybench_Plopper, oracles=dict(), **original_kwargs):
    if type(input_space_definition) is not CS.ConfigurationSpace:
        input_space_definition = BaseProblem.configure_space(input_space_definition)
    class Polybench_Problem(BaseProblem):
        input_space = input_space_definition
        parameter_space = None
        output_space = Space([Real(0.0, inf, name='time')])
        problem_params = dict((p.lower(), 'categorical') for p in input_space_definition.get_hyperparameter_names())
        categorical_cast = dict((p.lower(), 'str') for p in input_space_definition.get_hyperparameter_names())
        constraints = [ScalarRange(column_name='input', low_value=min(lookup.keys()), high_value=max(lookup.keys()), strict_boundaries=False)]
        dataset_lookup = lookup
        def __init__(self, class_size, **kwargs):
            # Allow anything to be overridden by passing it in as top priority
            for k, v in original_kwargs.items():
                kwargs.setdefault(k,v)
            expect_kwargs = {'use_capital_params': True,
                             'problem_class': class_size,
                             'dataset': f" -D{self.dataset_lookup[class_size][1]}_DATASET",
                             'plopper': plopper_class(there+"/mmp.c", there, output_extension='.c'),
                            }
            for k,v in expect_kwargs.items():
                kwargs.setdefault(k,v)
            super().__init__(**kwargs)
        def objective(self, point, *args, **kwargs):
            return super().objective(point, self.dataset, *args, **kwargs)
        def O3(self):
            # Temporarily swap references
            old_source = self.plopper.sourcefile
            self.plopper.sourcefile = self.name.split('_',1)[0].lower()+".c"
            rvalue = super().objective({}, self.dataset, O3=True)
            self.plopper.sourcefile = old_source
            return rvalue
    Polybench_Problem.__name__ = name
    inv_lookup = dict((v[0], k) for (k,v) in lookup.items())
    if default is None:
        default = inv_lookup['S']
    return import_method_builder(Polybench_Problem, inv_lookup, default, oracles)

