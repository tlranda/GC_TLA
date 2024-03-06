import platform
import subprocess
import shlex
import os
# Ensure that utilizing special (but reasonable) generic-type classes do not reject input
INTEGER_TYPES = [int]
STRING_TYPES = [str]
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

    def detect_number_of_nodes(self):
        self.nodes = 1
        if 'PBS_NODEFILE' in os.environ:
            with open(os.environ['PBS_NODEFILE'],'r') as f:
                self.nodes = len(f.readlines())

    def detect_machine_identifier(self):
        if 'HOSTNAME' in os.environ:
            self.machine_identifier = os.environ['HOSTNAME']
        else:
            self.machine_identifier = platform.node()

    def set_comparable(self):
        """
            If you subclass this class, must call this manually if you use super().__init__() first
        """
        self.comparable = [k for (k,v) in self.__dict__.items() if not k.startswith('_') and not callable(v) and k != 'machine_identifier']

    def __init__(self, threads_per_node=None,
                       gpus=None,
                       ranks_per_node=None,
                       nodes=None,
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
        # Get number of nodes
        if not self.default_assign('nodes', nodes, INTEGER_TYPES):
            self.detect_number_of_nodes()

        # DERIVE full MPI rank count
        self.mpi_ranks = self.nodes * self.ranks_per_node

        # Friendly name for this architecture
        if not self.default_assign('machine_identifier', machine_identifier, STRING_TYPES):
            self.detect_machine_identifier()

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

