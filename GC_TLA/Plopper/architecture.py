import platform
import subprocess
import shlex
import os
import pathlib
# Ensure that utilizing special (but reasonable) generic-type classes do not reject input
INTEGER_TYPES = [int]
STRING_TYPES = [str, pathlib.Path]
try:
    import numpy as np
    INTEGER_TYPES.append(np.integer)
    STRING_TYPES.append(np.bytes_)
except ImportError:
    pass
INTEGER_TYPES = tuple(INTEGER_TYPES)
STRING_TYPES = tuple(STRING_TYPES)

class Architecture():
    """
        Class to hold information about a platform architecture and automatically fetch it when not defined

        All assignable attributes should have a detect_ATTR() function to set their default
        All non-assignable attributes should derive their values after assignable values are initialized
    """
    def default_assign(self, attr, value, accepted_types):
        """
            Reject bad input, but do not error if value is None
        """
        if value is None:
            return False
        if not isinstance(value, accepted_types):
            raise TypeError(f"{attr} received type {type(value)}, must be in types: {accepted_types}")
        setattr(self, attr, value)
        return True

    def detect_threads_per_node(self):
        # Not always available, but directly returns # logical processors
        try:
            proc = subprocess.run(['nproc'], capture_output=True)
            OK = proc.returncode == 0
        except FileNotFoundError:
            OK = False
        if OK:
            self.threads_per_node = int(proc.stdout.decode('utf-8').strip())
        else:
            # LSCPU should detail logical number of processors
            try:
                proc = subprocess.run(['lscpu'], capture_output=True)
                for line in proc.stdout.decode('utf-8'):
                    if line.startswith('CPU(s)'):
                        self.threads_per_node = int(line.rstrip().rpslit(' ',1)[1])
                        break
            except FileNotFoundError:
                pass
            # Fail fast if neither worked
            if not hasattr(self, 'threads_per_node'):
                raise ValueError("Failed to determine threads_per_node; no default given")

    def detect_gpus(self):
        # Attempt to identify NVIDIA GPU count
        try:
            proc = subprocess.run(shlex.split('nvidia-smi -L'), capture_output=True)
            OK = proc.returncode == 0
        except FileNotFoundError:
            OK = False
        if OK:
            self.gpus = len(proc.stdout.decode('utf-8').strip().split('\n'))
        else:
            # May be on an AMD system instead
            try:
                proc = subprocess.run(shlex.split('rocm-smi -l'), capture_output=True)
                OK = proc.returncode == 0
            except FileNotFoundError:
                OK = False
            if OK:
                self.gpus = len(proc.stdout.decode('utf-8').strip().split('\n'))
            else:
                # Lacking nvidia-smi and rocm-smi leads one to believe that there are no GPUs
                self.gpus = 0

    def detect_ranks_per_node(self):
        if self.gpus > 0:
            self.ranks_per_node = self.gpus
        else:
            self.ranks_per_node = self.threads_per_node

    def detect_hostfile(self):
        self.hostfile = None
        if 'PBS_NODEFILE' in os.environ:
            self.hostfile = os.environ['PBS_NODEFILE']

    def detect_machine_identifier(self):
        if 'HOSTNAME' in os.environ:
            self.machine_identifier = os.environ['HOSTNAME']
        else:
            self.machine_identifier = platform.node()

    def set_comparable(self):
        """
            If you subclass this class, must call this manually if you use super().__init__() first
        """
        # Do not compare machine identifier as it may be customized without affecting architecture
        # Do not compare the comparable list itself
        incomparable = ['machine_identifier', 'comparable']
        # Do not compare under- or dunder- attributes and do not compare callable attributes
        incomparable.extend([k for (k,v) in self.__dict__.items() if k.startswith('_') or callable(v)])
        # Compare everything else
        self.comparable = [k for (k,v) in self.__dict__.items() if k not in incomparable]
        return self.comparable

    def get_derived(self):
        """
            Return attributes that are derived rather than set
        """
        comparable = set(self.comparable)
        assignable = set()
        for k in dir(self):
            if k.startswith('detect_') and callable(getattr(self, k)):
                # Drop detect_ prefix
                assignable.add(k[7:])
        return comparable.difference(assignable)

    def init_derivable(self, **kwargs):
        if 'nodes' in kwargs:
            raise ValueError("Nodes are derived from a hostfile, or 1 if no hostfile. Provide a hostfile with one line per host")
        if self.hostfile is None:
            self.nodes = 1
        else:
            with open(self.hostfile,'r') as f:
                self.nodes = len(f.readlines())

        if 'mpi_ranks' in kwargs:
            raise ValueError("MPI ranks are derived, set hostfile and ranks_per_node instead of directly setting MPI ranks")
        self.mpi_ranks = self.nodes * self.ranks_per_node

    def __init__(self, threads_per_node=None,
                       gpus=None,
                       ranks_per_node=None,
                       hostfile=None,
                       machine_identifier=None,
                       **kwargs):
        # Determine threads per node as 1 thread per logical processor
        if not self.default_assign('threads_per_node', threads_per_node, INTEGER_TYPES):
            self.detect_threads_per_node()
        # Determine number of GPUs per node
        if not self.default_assign('gpus', gpus, INTEGER_TYPES):
            self.detect_gpus()
        # Get number of ranks per node
        if not self.default_assign('ranks_per_node', ranks_per_node, INTEGER_TYPES):
            self.detect_ranks_per_node()
        # Get hostfile
        if not self.default_assign('hostfile', hostfile, STRING_TYPES):
            self.detect_hostfile()

        # Friendly name for this architecture
        if not self.default_assign('machine_identifier', machine_identifier, STRING_TYPES):
            self.detect_machine_identifier()

        # DERIVE full MPI rank count
        self.init_derivable(**kwargs)

        # Set comparable attributes
        self.set_comparable()

    def __str__(self):
        return f"Architecture[{self.machine_identifier}]:"+"{"+", ".join([f"{k}: {getattr(self,k)}" for k in self.comparable])+"}"

    def __eq__(self, other):
        if not isinstance(other, Architecture):
            return False
        # All attributes that are not the machine identifier should be equivalent
        for comp in self.comparable:
            if getattr(self, comp) != getattr(other, comp):
                return False
        return True

