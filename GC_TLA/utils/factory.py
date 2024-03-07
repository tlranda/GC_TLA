import pdb
import warnings
# Own library
from GC_TLA.utils import Configurable

class Factory(Configurable):
    def __init__(self, factory_class, factory_name=None, debug_class_construction=False,
                 initial_args=None, initial_args_first=True, initial_kwargs=None,
                 initial_configure=None, initial_direct=None):
        super().__init__()
        assert issubclass(factory_class, Configurable), "Factory classes must be GC_TLA Configurable subclasses"
        self.class_def = factory_class
        if factory_name is None:
            factory_name = factory_class.__name__
        self.class_name = factory_name
        self.debug_class_construction = debug_class_construction
        self.initial_args_first = initial_args_first
        # All of the following have non-None defaults, but use None for clarity
        if initial_direct is None:
            initial_direct = dict()
        self.direct_handles = initial_direct
        if initial_args is None:
            initial_args = list()
        self.pass_on_init_args = initial_args
        if initial_kwargs is None:
            initial_kwargs = dict()
        self.pass_on_init_kwargs = initial_kwargs
        # Call the configure method if initial decision is ready
        if initial_configure is not None:
            self.configure_class(**initial_configure)

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
            if len(self.pass_on_init_args) > 0 and warn_extension:
                warnings.warn(f"Did not match {len(args[arg_idx:])} new arguments, extending argument list", UserWarning)
            self.pass_on_init_args.extend(args[arg_idx:])

    def configure_class(self, **kwargs):
        # Update the configurable core of the FactoryConfigurable class --
        # changes here will propagate to ALL subsequent instantiations
        self.class_def._configure(**kwargs)

    def map_identifier_to_args(self, identifier):
        # Default implementation returns the identifier as a single argument without any kwargs
        # When overriding, always return a tuple of:
        #   1) list of positional args (that come AFTER pass_on_init args)
        #   2) dictionary with additional keyword arguments
        return ([identifier], dict())

    def handles(self, name):
        # Name will be <class_name>_<problem_identifier>
        # But for extensibility, we map problem identifier to args/kwargs
        if '_' not in name:
            return (False, list(), dict())
        name, identifier = name.split('_',1)
        if name == self.class_name:
            return (True, *self.map_identifier_to_args(identifier))
        else:
            return (False, list(), dict())

    def build(self, name, *args, **kwargs):
        if self.initial_args_first:
            init_args = self.pass_on_init_args + list(args)
        else:
            init_args = list(args) + self.pass_on_init_args
        init_kwargs = dict((k,v) for (k,v) in self.pass_on_init_kwargs.items())
        init_kwargs.update(**kwargs)
        # Weird edge case: if both pass_on_init kwargs and kwargs are empty, this becomes None
        if init_kwargs is None:
            init_kwargs = dict()
        instance = self.class_def(*init_args, **init_kwargs)
        return instance

    def getattr(self, name):
        if self.debug_class_construction:
            pdb.set_trace()
        if name in self.direct_handles:
            return self.direct_handles[name]
        (buildable, args, kwargs) = self.handles(name)
        if buildable:
            return self.build(name, *args, **kwargs)
        return

