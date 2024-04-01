import subprocess
# Own library
from GC_TLA.plopper import (Arch, Executor)
from GC_TLA.utils import Configurable

class EphemeralPlopper(Configurable):
    def __init__(self, *args, force_write=False, base_substitution=None,
                 executor=None, architecture=None, **kwargs):
        super().__init__()

        self.force_write = force_write
        if base_substitution is None:
            base_substitution = dict()
        self.base_substitution = base_substitution

        if executor is not None and not isinstance(executor, Executor):
            raise TypeError(f"Type executor ({type(executor)}) is not a GC_TLA Executor")
        self.executor = executor

        if architecture is not None and not isinstance(architecture, Arch):
            raise TypeError(f"Type architecture ({type(architecture)}) is not a GC_TLA Architecture")
        elif architecture is None:
            architecture = Arch() # Default system detection
        self.architecture = architecture

        # Str-able attributes list
        self.str_attrs = ['force_write', 'executor', 'architecture']
        self.str_objs = ['force_write']

    def __str__(self):
        rstring = f"{self.__class__.__name__}"+"{"
        str_attrs = []
        for attr in self.str_attrs:
            if attr in self.str_objs:
                str_attrs.append(f"{attr}: {getattr(self,attr)}")
            else:
                str_attrs.append(f"{attr}: {getattr(self,attr).__repr__()}")
        rstring += ",\n".join(str_attrs)+"}"
        return rstring

    def buildTemplateCmds(self, outfile, *args, **kwargs):
        """
            Return None to skip executing any commands after templating a file
            Otherwise, return a list of string commands to execute via the Python3 subprocess library
        """
        return None

    def buildExecutorCmds(self, outfile, *args, **kwargs):
        """
            Return None to skip executing any commands after templating a file
            Otherwise, return a list of string commands to execute for evaluation via Executor instance
        """
        return None

    def setDestination(self, destination=None, use_raw_template=False, *args, **kwargs):
        """
            Alter destination based on use_raw_template
            Recommended for subclases to convert destination=None into a unique and valid path
        """
        return destination

    def fillTemplate(self, destination, *args, **kwargs):
        """
            Create a file at destination and fill it with content
        """
        return

    def templateExecute(self, destination=None, *args, use_raw_template=False, **kwargs):
        destination = self.setDestination(destination, use_raw_template, *args, **kwargs)
        # May be useful for unit-testing to know where the previous output was placed
        self.previous_destination = destination

        # Produce pre-execution commands
        template_cmds = self.buildTemplateCmds(destination, *args, **kwargs)
        if not use_raw_template and (self.force_write or template_cmds is not None):
            self.fillTemplate(destination, *args, **kwargs)

        if template_cmds is not None:
            env = None if self.executor is None else self.executor.set_os_environ(None)
            for cmd_i, cmd in enumerate(template_cmds):
                status = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, env=env)
                if status.returncode != 0:
                    print(f"Template command Failed: {cmd}")
                    print(status.stderr)
                    if self.executor is None:
                        return
                    return self.executor.unable_to_execute()
        # Use executor to determine metrics of interest
        if self.executor is None:
            return
        else:
            return self.executor.execute(destination, self.buildExecutorCmds, *args, **kwargs)

