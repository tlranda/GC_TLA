import pdb
# Dependent modules
import ConfigSpace as CS
# Own library
from GC_TLA.Problem import Problem_Classes
from GC_TLA.Plopper.architecture import Architecture as Arch
from GC_TLA.Plopper.executor import Executor
from GC_TLA.Plopper.plopper import Plopper


class ProblemFactory():
    def __init__(self, factory_class, debug_class_construction=False, arch=None, executor=None, plopper=None, plopper_template=None):
        self.class_def = factory_class
        self.class_name = factory_class.__name__
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

    def configure_for_class_constructor(self, *args, **kwargs):
        self.pass_on_init_args = args
        self.pass_on_init_kwargs.update(kwargs)

    def handles(self, name):
        # Name will be <class_name>_<problem_identifier>
        if '_' not in name:
            return (False, None)
        name, identifier = name.split('_',1)
        if name == self.class_name:
            return (True, identifier)
        else:
            return (False, None)

    def build(self, name, identifier):
        init_args = self.pass_on_init_args + [identifier]
        if self.debug_class_construction:
            pdb.set_trace()
        instance = self.class_def(*init_args, **self.pass_on_init_kwargs)
        return instance

    def getattr(self, name):
        (handled, identifier) = self.handles(name)
        if handled:
            return self.build(name, identifier)
        return
