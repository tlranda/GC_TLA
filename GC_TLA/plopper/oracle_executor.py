import pathlib
# DEPENDENT MODULES
import numpy as np
import pandas as pd
# Own library
from GC_TLA.plopper import MetricIDs, Executor

class OracleExecutor(Executor):
    """
        When evaluation results are already known, can utilize an OracleEvaluator to look up values in oracleSearch()
    """
    def __init__(self, oracle_path=None, oracle_sort_keys=None, oracle_match_cols=None, oracle_return_cols=None,
                 # Settings for parent class
                 evaluation_tries=1, retries=0, infinity=None,
                 ignore_runtime_failures=False, timeout=None, strict_cleanup=False):
        super().__init__(evaluation_tries, retries, infinity, ignore_runtime_failures, timeout, strict_cleanup)
        # When oracle isn't provided, fall back to operating as a standard executor
        if oracle_path is None:
            self.as_oracle = False
            return
        self.as_oracle = True
        self.oracle = pathlib.Path(oracle_path)
        self.oracle_data = pd.read_csv(self.oracle)
        if oracle_sort_keys is not None:
            self.oracle_data = self.oracle_data.sort_values(by=oracle_sort_keys)
        if oracle_match_cols is None:
            oracle_match_cols = list(self.oracle_data.columns)
        self.oracle_match_cols = oracle_match_cols
        self.oracle_match_n = len(self.oracle_match_cols)
        self.oracle_matching = self.oracle_data[self.oracle_match_cols].astype(str)
        if oracle_return_cols is None:
            oracle_return_cols = list(self.oracle_data.columns)
        self.oracle_return_cols = oracle_return_cols

    def oracleSearch(self, search, *args, as_rank=False, single_return=False, **kwargs):
        """
            search should be matchable to an entire record in the oracle_match_cols subset of oracle data
            when as_rank is True, returns the oracle rank rather than the indicated oracle_return_cols
            when single_return is True, ensures only a single result is returned
                Typically, this means that you get a scalar for as_rank=True or a pd.Series if as_rank=False
                This distinction is expected to matter most for produceMetric() implementation
        """
        if not self.as_oracle:
            raise ValueError("Not initialized with an oracle! Only able to operate as standard executor!")
        if type(search) is not tuple:
            search = tuple(search)
        n_matching_columns = (self.oracle_matching == search).sum(1)
        full_match_idx = np.nonzero(n_matching_columns == self.oracl_match_n)[0]
        if len(full_match_idx) == 0:
            raise ValueError(f"No complete matches for {search} in oracle {self.oracle}")
        if as_rank:
            if single_return:
                return full_match_idx[0]
            else:
                return full_match_idx
        if single_return:
            return self.produceMetric(self.oracle_data.loc[full_match_idx[0], self.oracle_return_cols].values)
        else:
            return self.produceMetric(self.oracle_data.loc[full_match_idx, self.oracle_return_cols])

