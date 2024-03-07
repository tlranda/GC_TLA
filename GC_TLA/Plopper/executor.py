import subprocess
import pathlib
import time
import warnings
import enum
# NON-DEFAULT MODULES
import numpy as np
# Own library
from GC_TLA.Utils import Configurable

# ENUM for infinity values -- used by implementers and users!
class MetricIDs(enum.Enum):
    """
        Use as keys for dictionary to customize the 'infinity' argument of Executors
        Specifying {MetricIDs.NotOK: <VALUE>} is minimally sufficient
    """
    OK = enum.auto() # Use parsed metric
    NotOK = enum.auto() # Generic catch-all when parsed metric is unavailable
    TimeOut = enum.auto() # Metric unavailable due to timeout
    BadReturnCode = enum.auto() # Metric unavailable due to bad return code
    BadParse = enum.auto() # Metric unavailable due to failure in metric parsing
    UnableToExecute = enum.auto() # Metric unavailable due to failure PRIOR to execution (compilation error, etc)

    @classmethod
    def validate_infinity_mapping(cls, metric_dict):
        # As long as NotOK is available, we're good
        if cls.NotOK in metric_dict.keys():
            return True
        # Else you have to implement all other NotOK enumerations
        all_keys = list(cls.__members__.keys())
        all_values = list(cls.__members__.values())
        other_notOK = all_values[all_keys.index(cls.NotOK.name)+1:]
        for name in other_notOK:
            if name not in metric_dict.keys():
                return False
        return True

class Executor(Configurable):
    """
        Executes runtime-ready shell code via subprocess, utilizing aggregation metrics
        and limited retries to get stable measurements
    """
    def __init__(self, evaluation_tries=3, retries=0, infinity=None,
                 ignore_runtime_failure=False, timeout=None, strict_cleanup=False):
        super().__init__()
        self.evaluation_tries = evaluation_tries
        self.retries = retries
        if infinity is None:
            infinity = {MetricIDs.NotOK: np.inf}
        elif not MetricIDs.validate_infinity_mapping(infinity):
            raise ValueError(f"Supplied infinity mapping '{infinity}' missing NotOK key and one or more specific failure keys")
        self.infinity = infinity
        self.ignore_runtime_failure = ignore_runtime_failure
        self.timeout = timeout
        self.strict_cleanup = strict_cleanup

    def getMetric(self, logfile, outfile, attempt, *args, aggregator_fn=None, **kwargs):
        """
            Very simple parsing expects metric to be a numeric value (or newline-delimited series of values to interpret via aggregator_fn)
        """
        if logfile is None:
            return None
        with open(logfile, 'r') as f:
            data = [_.rstrip() for _ in f.readlines()]
            if len(data) == 1:
                try:
                    return float(data[0])
                except ValueError:
                    return None
            else:
                try:
                    total = [float(x) for x in data]
                    return aggregator_fn(total)
                except (ValueError, TypeError):
                    return None

    def produceMetric(self, metric_list):
        # Allows for different interpretations of repeated events
        # Defaults to optimize for minimum value
        return min(metric_list)

    def unable_to_execute(self):
        """
            Helper function to build appropriate metric value when .execute() cannot be called
        """
        if MetricIDs.UnableToExecute in self.infinity.keys():
            unable_metric = [self.infinity[MetricIDs.UnableToExecute]]
        else:
            unable_metric = [self.infinity[MetricIDs.NotOK]]
        return self.produceMetric(unable_metric * self.evaluation_tries)

    def set_os_environ(self, attempt):
        """
            Return a mapping of environment variables for each trial's process
            Attempt = None if this is a template command, else # of execution trial
        """
        return None

    def cleanup(self, outfile, attempt):
        # Perform any post-execution actions to ensure the next execution will be capable of running properly
        return

    def sufficiently_logged(self, cmd_queue, last_idx, timeouts):
        """
            Determine if last executed / timed out command is likely to have done enough to produce
            usable logs
            Subclasses can update if analysis for the metric is not the final command in the queue
        """
        return last_idx >= len(cmd_queue)-1

    def execute(self, outfile, runstr_fn, *args, **kwargs):
        """
        Using outfile and the attempt # (as well as *args, **kwargs),
        runstr_fun() should return a list of subprocess-ready strings to execute
        This function will attempt to execute each string with retries and return the metric, ergo
        it is the primary method for other classes to interact with
        """
        metrics = []
        failures = 0
        attempt = 0
        while failures <= self.retries and len(metrics) < self.evaluation_tries:
            run_strs = runstr_fn(outfile, attempt, *args, **kwargs)
            env = self.set_os_environ(attempt)
            out, errs = None, None
            logged = False
            # This is OK even if outfile is pathlib.Path already
            logfile = pathlib.Path(outfile)
            # .with_stem only available in Python 3.9+
            logfile = logfile.with_name(f"{logfile.stem}_{attempt}.log")

            # Beyond this point, attempt is updated -- things depending on attempt should be set above
            attempt += 1
            with open(logfile, "w") as logs:
                timeouts = False
                trial_start_time = time.time()
                if self.timeout is not None:
                    for cmd_i, r_str in enumerate(run_strs):
                        execution_status = subprocess.Popen(r_str, shell=True, stdout=logs, stderr=logs, env=env)
                        #child_pid = execution_status.pid
                        try:
                            execution_status.communicate(timeout=self.timeout)
                        except subprocess.TimeoutExpired:
                            try:
                                timeouts = True
                                execution_status.kill()
                                if not self.ignore_runtime_failure:
                                    break
                            except ProcessLookupError:
                                # Sometimes the process ends quickly/slowly and we can generate this exception
                                if not self.ignore_runtime_failure:
                                    break
                            time.sleep(1)
                else:
                    for cmd_i, r_str in enumerate(run_strs):
                        execution_status = subprocess.run(r_str, shell=True, stdout=logs, stderr=logs, env=env)
                        if not self.ignore_runtime_failure and execution_status.returncode != 0:
                            break
                trial_duration = time.time() - trial_start_time
            # Determine whether the logs are expected to be useful or not
            logged = self.sufficiently_logged(run_strs, cmd_i, timeouts)
            # Cleanup may be defined between executions
            try:
                self.cleanup(outfile, attempt)
            except Exception as e:
                # Exceptions should produce warnings, not crash the program unless indicated to be strict
                if self.strict_cleanup:
                    raise e
                BadCleanup = f"Bad cleanup ({e.__class__})"
                for attr in ['msg','message','args']:
                    if hasattr(e, attr):
                        BadCleanup += f" -- {attr}:{geattr(e,attr)}"
                warnings.warn(BadCleanup)
            if logged and not self.ignore_runtime_failure and execution_status.returncode != 0:
                # FAILURE
                failures += 1
                run_failed = f"FAILED EXECUTION: '{run_str}' -- return code {execution_status.returncode}"
                warnings.warn(run_failed)
                if MetricIDs.BadReturnCode in self.infinity.keys():
                    metrics.append(self.infinity[MetricIDs.BadReturnCode])
                else:
                    metrics.append(self.infinity[MetricIDs.NotOK])
                continue
            # Find the execution time
            elif logged:
                logged_metric = self.getMetric(logfile, outfile, attempt-1, *args, **kwargs)
                if logged_metric is not None:
                    # MetricIDs.OK
                    metrics.append(logged_metric)
                else:
                    bad_logs_warning = f"Failed to read {logfile}"
                    warnings.warn(bad_logs_warning)
                    failures += 1
                    if MetricIDs.BadParse in self.infinity.keys():
                        metrics.append(self.infinity[MetricIDs.BadParse])
                    else:
                        metrics.append(self.infinity[MetricIDs.NotOK])
            else:
                # Timed out evaluations MAY be recoverable
                derived_timeout = self.getMetric(logfile, outfile, attempt-1, *args, **kwargs)
                if derived_timeout is None:
                    failures += 1
                    timeout_warning = f"FAILED EXECUTION: '{run_str}' -- timed out; non-recoverable"
                    warnings.warn(timeout_warning)
                    if MetricIDs.TimeOut in self.infinity.keys():
                        metrics.append(self.infinity[MetricIDs.TimeOut])
                    else:
                        metrics.append(self.infinity[MetricIDs.NotOK])
                else:
                    # Recoverable! MetricIDs.OK
                    metrics.append(derived_timeout)
        # Unable to evaluate this execution
        if failures > self.retries:
            print(f"OVERALL FAILED: {run_str}")
        return self.produceMetric(metrics)

