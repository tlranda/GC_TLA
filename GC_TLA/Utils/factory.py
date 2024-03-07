import pdb
import warnings
# Dependent modules
import ConfigSpace as CS
# Own library
from GC_TLA.Utils.factory_configurable import FactoryConfigurable
from GC_TLA.Problem import Problem_Classes
from GC_TLA.Plopper.architecture import Architecture as Arch
from GC_TLA.Plopper.executor import Executor
from GC_TLA.Plopper.plopper import Plopper

class Factory(FactoryConfigurable):
    def __init__(self, factory_class, debug_class_construction=False, arch=None, executor=None, plopper=None, plopper_template=None):
        assert issubclass(factory_class, FactoryConfigurable), "Factory classes must be GC_TLA FactoryConfigurable subclasses"
        self.class_def = factory_class
        self.class_name = factory_class.__name__
        self.direct_handles = dict()
        if arch is None:
            arch = Arch()
        if executor is None:
            executor = Executor()
        if plopper is None:
            if plopper_template is None:
                plopper_template = 'dum'
            plopper = Plopper(plopper_template, executor=executor, architecture=arch)
        self.pass_on_init_args = [arch, executor, plopper, CS.ConfigurationSpace(),]
        self.pass_on_init_kwargs = dict()
        self.debug_class_construction = debug_class_construction

    def update_init_args(self, *args, ignore_type_match=False, warn_extension=True):
        arg_idx = 0
        for idx in range(len(self.pass_on_init_args)):
            if type(self.pass_on_init_args[idx]) is not type(args[arg_idx]):
                if ignore_type_match:
                    self.pass_on_init_args[idx] = args[arg_idx]
                    arg_idx += 1
                    if arg_idx >= len(args):
                        break
                else:
                    warnings.warn(f"Stored Argument #{idx} has type {type(self.pass_on_init_args[idx])}, does not match new argument #{arg_idx} type {type(args[arg_idx])}. Skipping for exact type match", UserWarning)
            else:
                self.pass_on_init_args[idx] = args[arg_idx]
                arg_idx += 1
                if arg_idx >= len(args):
                    break
        if len(args[arg_idx:]) > 0:
            if warn_extension:
                warnings.warn(f"Did not match {len(args[arg_idx:])} new arguments, extending argument list", UserWarning)
            self.pass_on_init_args.extend(args[arg_idx:])

    def configure_class(self, **kwargs):
        # Update the configurable core of the FactoryConfigurable class --
        # changes here will propagate to ALL subsequent instantiations
        self.class_def._configure(**kwargs)

    def handles(self, name):
        # Name will be <class_name>_<problem_identifier>
        if '_' not in name:
            return (False, False, None)
        name, identifier = name.split('_',1)
        if name == self.class_name:
            return (False, True, identifier)
        else:
            return (False, False, None)

    def build(self, name, identifier):
        init_args = self.pass_on_init_args + [identifier]
        if self.debug_class_construction:
            pdb.set_trace()
        instance = self.class_def(*init_args, **self.pass_on_init_kwargs)
        return instance

    def getattr(self, name):
        if name in self.direct_handles:
            return self.direct_handles[name]
        (buildable, value) = self.handles(name)
        if buildable:
            return self.build(name, value)
        return

