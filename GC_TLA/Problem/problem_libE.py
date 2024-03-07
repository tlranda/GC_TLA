import time
# Dependent modules
from sdv.constraints import ScalarRange
# Own library
from GC_TLA.Problem.problem import Problem
from GC_TLA.Utils.param_space import ParamSpace, Real, Integer, Categorical, inf

class LibEProblem(Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_space = ParamSpace([Real(0.0, inf, name='time')])
        self.constraints = [ScalarRange(column_name='input', low_value=1, high_value=128, strict_boundaries=False)]
        # Lookup to be added by subclasses

    def _log(self, config, result):
        if self.logfile is None:
            raise ValueError("No logfile configured")
        # Match the format made by LibEnsemble logs
        alter_config = dict((k,v) for (k,v) in config.items() if k not in ['machine_info',])
        alter_config[self.result_col] = result
        alter_config[self.time_col] = time.time() - self.start_time
        alter_config['libE_id'] = 0
        from_machine_info = dict((k,config['machine_info'][k]) for k in ['libE_workers','identifier','mpi_ranks','threads_per_node','ranks_per_node','gpu_enabled'])
        alter_config.update(from_machine_info)
        frame = pd.DataFrame(data=[alter_config], columns=list(alter_config.keys()))
        if self.logfile.exists():
            frame = pd.read_csv(self.logfile).append(frame, ignore_index=True)
        frame.to_csv(self.logfile, index=False)

