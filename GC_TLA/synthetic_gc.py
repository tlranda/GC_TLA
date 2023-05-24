import pdb
import warnings
import numpy as np, pandas as pd, math
import os, argparse, inspect
from csv import writer
from ytopt.search.util import load_from_file

# Models
from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE
# Will only use one of these, make a selection dictionary
sdv_models = {'GaussianCopula': GaussianCopula,
              'CopulaGAN': CopulaGAN,
              'CTGAN': CTGAN,
              'TVAE': TVAE,
              'random': None}
def check_conditional_sampling(objectlike):
    # Check source code on the _sample() method for the NotImplementedError
    # Not very robust, but relatively lightweight check that should be good enough
    # to find SDV's usual pattern for models that do not have conditional sampling yet
    try:
        source = inspect.getsource(objectlike._sample)
    except AttributeError:
        if objectlike is not None:
            print(f"WARNING: {objectlike} could not determine conditional sampling status")
        return False
    return not("raise NotImplementedError" in source and
               "doesn't support conditional sampling" in source)
conditional_sampling_support = dict((k,check_conditional_sampling(v)) for (k,v) in sdv_models.items())
from sdv.constraints import CustomConstraint, Between
from sdv.sampling.tabular import Condition

# Budget hypergeometrics
try:
    from math import comb
except ImportError:
    from math import factorial
    def comb(n,k):
        return factorial(n) / (factorial(k) * factorial(n-k))
def hypergeo(i,p,t,k):
    return (comb(i,t)*comb((p-i),(k-t))) / comb(p,k)

# Can only use exhaustive data where it exists, determine which ones we have at runtime
from problem import oracles, lookup_ival
exhausts = list(oracles.keys())
# Convert lookup_ival into presentable sizes
inverse_lookup = dict((v[0],k) for (k,v) in lookup_ival.items())
sources = list(inverse_lookup.keys())

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='RNG seed')
    # Options that control what data is generated and how
    fitting = parser.add_argument_group('fitting')
    fitting.add_argument('--sources', type=str, choices=sources, nargs="+", required=True, help="Input sizes to present as fitting data")
    fitting.add_argument('--exhaust', type=str, choices=exhausts, default=exhausts[0], help="Size to predict in fewshot based on generated data")
    fitting.add_argument('--initial-samples', nargs="+", type=int, default=None, help="Number of initial samples per input size (default 200)")
    fitting.add_argument('--min-quantile', dest='min', nargs="+", type=float, default=None, help="Minimum quantile to sample (default 0.0)")
    fitting.add_argument('--max-quantile', dest='max', nargs="+", type=float, default=None, help="Maximum quantile to sample (default 1.0)")
    fitting.add_argument('--mean-quantile', dest='mean', nargs="+", type=float, default=None, help="Mean quantile to sample (default 0.5)")
    #fitting.add_argument('--stddev-quantile', dest='stddev', nargs="+", type=float, default=None, help="Standard deviation to sample (default 1.0)")
    # Options that control how the model is built and evaluated
    fewshot = parser.add_argument_group('fewshot')
    fewshot.add_argument('--max-evals', type=int, default=30, help='Maximum number of samples to generate (default %(default)s)')
    fewshot.add_argument('--top-fit', type=float, default=0.3, help='Top proportion of "sampled" data to fit to (default %(default)s)')
    fewshot.add_argument('--model', choices=list(sdv_models.keys()), default='GaussianCopula', help='SDV model (default %(default)s)')
    # Options to control how budget is determined
    budget = parser.add_argument_group('budget')
    budget.add_argument('--budget', type=int, default=None, help="Supply arbitrary budget instead of determining one")
    budget.add_argument('--ideal-fit', dest='ideal', type=float, default=0.1, help="Quantile to ideally sample in fewshot (default %(default)s)")
    budget.add_argument('--error-rate', dest='exclude', type=float, default=0.05, help="Proprotion of model space reduction that may have been optimal and is no longer sampleable (default %(default)s)")
    budget.add_argument('--p-success', dest='success_bar', type=float, default=0.9, help="Target likelihood of fewshot success for budget tuning (default %(default)%)")
    return parser

# This class is a dumb workaround to let me 'show' where the oracle data comes from without having the entire
# DataFrame print if you look at the args Namespace object
class oraclePandasWrapper(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ORACLE_FILENAME = None
    def __repr__(self):
        return self.ORACLE_FILENAME if self.ORACLE_FILENAME is not None else super().__str__()

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # Load oracle data here
    args.oracle = oraclePandasWrapper(pd.read_csv(oracles[args.exhaust]).sort_values(by='objective').reset_index(drop=True))
    args.oracle.ORACLE_FILENAME=oracles[args.exhaust]
    n_sampleable = len(args.oracle)
    # Set numpy seed during parsing
    np.random.seed(args.seed)
    # Sanity for other things
    if args.ideal < 0 or args.ideal > 1:
        raise ValueError(f"Ideal quantile (--ideal-fit {args.ideal}) must be between 0 and 1")
    if args.success_bar < 0 or args.success_bar > 1:
        raise ValueError(f"Probability of fewshot success (--p-success {args.success_bar}) must be between 0 and 1")
    if args.exclude < 0 or args.exclude >= 1:
        raise ValueError(f"Model space reduction (--error-rate {args.exclude}) must be >= 0 and < 1")
    # Defaults
    if args.initial_samples is None:
        args.initial_samples = 200
    if args.min is None:
        args.min = 0.0
    if args.max is None:
        args.max = 1.0
    if args.mean is None:
        args.mean = 0.5
    #if args.stddev is None:
    #    args.stddev = 1.0
    if type(args.initial_samples) is not list:
        args.initial_samples = [args.initial_samples]
    if type(args.min) is not list:
        args.min = [args.min]
    if type(args.max) is not list:
        args.max = [args.max]
    if type(args.mean) is not list:
        args.mean = [args.mean]
    #if type(args.stddev) is not list:
    #    args.stddev = [args.stddev]
    # Extend quantile data based on sources (repeat last specified),
    # also perform sanity checks
    n_sources = len(args.sources)
    for idx in range(n_sources):
        if len(args.initial_samples)-1 < idx:
            args.initial_samples.append(args.initial_samples[-1])
        if args.initial_samples[idx] < 1:
            raise ValueError(f"Initial samples ({idx}th --initial-samples {args.initial_samples[idx]}) must be > 1")
        if len(args.min)-1 < idx:
            args.min.append(args.min[-1])
        if args.min[idx] < 0 or args.min[idx] > 1:
            raise ValueError(f"Minimum quantile ({idx}th --min-quantile {args.min[idx]}) must be between 0 and 1")
        if len(args.max)-1 < idx:
            args.max.append(args.max[-1])
        if args.max[idx] < 0 or args.max[idx] > 1:
            raise ValueError(f"Maximum quantile ({idx}th --max-quantile {args.max[idx]}) must be between 0 and 1")
        if len(args.mean)-1 < idx:
            args.mean.append(args.mean[-1])
        if args.mean[idx] < 0 or args.mean[idx] > 1:
            raise ValueError(f"Mean quantile ({idx}th --mean-quantile {args.mean[idx]}) must be between 0 and 1")
        #if len(args.stddev)-1 < idx:
        #    args.stddev.append(args.stddev[-1])
        # Sanity checks on FITTING
        if args.min[idx] >= args.mean[idx]:
            raise ValueError(f"Minimum quantile ({idx}th --min-quantile {args.min[idx]}) must be < mean ({idx}th --mean-quantile {args.mean[idx]})")
        if args.mean[idx] >= args.max[idx]:
            raise ValueError(f"Mean quantile ({idx}th --mean-quantile {args.mean[idx]}) must be < max ({idx}th --max-quantile {args.max[idx]})")
        if n_sampleable * (args.max[idx]-args.min[idx]) < args.initial_samples[idx]:
            print(f"!! Warning !! {idx}th sample range (max-min == {args.max[idx]}-{args.min[idx]} ==> {n_sampleable*(args.max[idx]-args.min[idx])}) smaller than requested samples (--initial-samples {args.initial_samples[idx]})")
            args.initial_samples[idx] = args.max[idx]-args.min[idx]
    return args

def generate_dataset(args):
    dataset = []
    ground_truth = []
    for source, samples, rmin, rmax, rmean in zip(args.sources, args.initial_samples, args.min, args.max, args.mean):
        # Allocate sampling mass divided by mean (bias +1 to left if necessary)
        n_sampleable = len(args.oracle)
        right_samples = n_sampleable * (rmax-rmean)
        mod = math.ceil(right_samples) - int(right_samples)
        right_samples = int(right_samples)
        left_samples = math.ceil(n_sampleable * (rmean-rmin)) + mod
        # Describe the sampling probability
        # For now it is uniform, in the future it'd be good to use stddev or something to have non-uniform distribution
        prob = [0 for _ in range(0, int(n_sampleable*rmin))] +\
               [0.5/left_samples for _ in range(int(n_sampleable*rmin), int(n_sampleable*rmean))] +\
               [0.5/right_samples for _ in range(int(n_sampleable*rmean), int(n_sampleable*rmax))] +\
               [0 for _ in range(int(n_sampleable*rmax), n_sampleable)]
        # Fix single element in mean (should be MINOR deviation) for probability sum to be exactly 1.0
        prob[int(n_sampleable*rmean)] -= sum(prob)-1.0
        # Take samples using choice
        idxs = np.sort(np.random.choice(n_sampleable, samples, replace=False, p=prob))
        ground_truth.append(idxs)
        # Build the 'source task dataset'
        task = args.oracle.iloc[idxs]
        task.insert(0,'input',[inverse_lookup[source]] * len(task))
        dataset.append(task)
    return pd.concat(dataset), np.stack(ground_truth)

def sample_approximate_conditions(model_name, model, conditions, criterion, param_names):
    if conditional_sampling_support[model_name]:
        return model.sample_conditions(conditions)
    # Reject-sampling with conditional enforcement
    requested_rows = sum([_.get_num_rows() for _ in conditions])
    selected = []
    prev_len = -1
    cur_len = 0
    while prev_len < requested_rows and cur_len != prev_len:
        prev_len = cur_len
        samples = model.sample(num_rows = requested_rows, randomize_samples=False)
        selected.append(pd.concat(samples))
        cur_len = sum(map(len, selected))
    selected = pd.concat(selected)
    # FORCE conditions to be held in data
    for cond in conditions:
        for (col, target) in cond.get_column_values().items():
            selected[col] = target
    return selected

def reidentify(frame, param_names, args):
    oracle = args.oracle[param_names].astype(str).to_numpy()
    n_params = len(param_names)
    ranks = []
    for idx, row in frame.iterrows():
        # Find the index of a full match of tuple values for all parameters
        # Since oracle is sorted, this is its global rank
        row_rank = np.where((oracle == tuple(row.to_list()[1:])).sum(1) == n_params)[0][0]
        ranks.append(row_rank)
    frame.insert(0, 'rank', ranks)
    return frame

def get_budget(model, conditions, problem, args):
    generated = []
    param_names = problem.params
    dup_cols = ['input']+param_names
    # Generation based on up to once per sample (permit uniform distribution)
    while sum(map(len,generated)) < problem.input_space_size:
        warnings.simplefilter('ignore')
        sampled = sample_approximate_conditions(args.model, model, conditions, [inverse_lookup[args.exhaust]], param_names)
        for col in param_names:
            sampled[col] = sampled[col].astype(str)
        warnings.simplefilter('default')
        sampled = sampled[dup_cols]
        generated.append(sampled)
    # Trim dataset
    generated = pd.concat(generated).iloc[:len(args.oracle)].drop_duplicates()
    # Re-identify ranks
    generated = reidentify(generated, param_names, args)
    # If arbitrary budget, just return the set of samples
    if args.budget is not None:
        return args.budget, generated
    # Analysis to determine budget
    Initial_I = int(problem.input_space_size * args.ideal)
    C = len(generated)
    Reduce_I = int(args.exclude * (problem.input_space_size - C))
    I_Prime = max(1, Initial_I - Reduce_I)
    if I_Prime > C:
        print(f"Ideal population smaller than biased population, budget indeterminate")
        return None, generated
    k = 0
    while k < I_Prime:
        k += 1
        confidence = sum([hypergeo(I_Prime,C,_,k) for _ in range(1,k+1)])
        if confidence >= args.success_bar:
            break
    if k < I_Prime or confidence >= args.success_bar:
        print(f"Reached {args.success_bar} probability with budget of {k} samples")
        return k, generated
    else:
        print(f"Budget failed to reach target probability; maximum {100*confidence:.2f}% probability with {k} samples")
        return None, generated

def main(args=None, prs=None):
    if prs is None:
        prs = build()
    args = parse(prs, args)
    print(args)
    # Generate dataset
    fittable, ground_truth = generate_dataset(args)

    # Load useful data from problem instance, even though we will NOT use it to evaluate
    problem = load_from_file('problem', args.exhaust)
    param_names = problem.params
    constraints = problem.constraints
    field_names = ['input']+param_names+['objective']
    field_transformers = problem.problem_params
    model = sdv_models[args.model](field_names=field_names, field_transformers=field_transformers, constraints=constraints, min_value=None, max_value=None)
    conditions = [Condition({'input': inverse_lookup[args.exhaust]}, num_rows=max(100, args.max_evals))]
    budget_conditions = [Condition({'input': inverse_lookup[args.exhaust]}, num_rows=problem.input_space_size)]
    # Fit data
    warnings.simplefilter('ignore')
    model.fit(fittable)
    warnings.simplefilter('default')

    # Determine the budget for this tuner and utilize it
    args.budget, sampled = get_budget(model, budget_conditions, problem, args)
    max_samples = args.max_evals if args.budget is None else min(args.max_evals, args.budget)
    # Determine success from samples
    within_budget = sampled.iloc[:max_samples]
    possible_ideal = int(problem.input_space_size*args.ideal)
    print(f"# of ideal configurations: {possible_ideal}")
    ever_ideal = np.where(sampled['rank'] <= possible_ideal)[0]
    if len(ever_ideal) == 0:
        print(f"NEVER samples ideal configurations")
    else:
        print(f"# ever sampled in ideal: {len(ever_ideal)} (up to {ever_ideal[-1]}/{len(sampled)} samples)")
        # Report rank as +1 to user as they will expect 1-indexing
        print("\t"+f"Best EVER sample: {1+min(sampled['rank'])} at sample {np.argmin(sampled['rank'])}")
        budget_ideal = np.where(within_budget['rank'] <= possible_ideal)[0]
        if len(budget_ideal) == 0:
            print(f"FAILED to sample ideal within budget")
        else:
            print(f"BUDGETED # sampled in ideal: {len(budget_ideal)} (early as {budget_ideal[0]}/{max_samples}; late as {budget_ideal[-1]}/{max_samples})")
            # Report rank as +1 to user as they will expect 1-indexing
            print("\t"+f"Best BUDGET sample: {1+min(within_budget['rank'])} at sample {np.argmin(within_budget['rank'])}")
        if max_samples < args.max_evals:
            within_max = sampled.iloc[:args.max_evals]
            max_ideal = np.where(within_max['rank'] <= possible_ideal)[0]
            if len(max_ideal) == 0:
                print(f"FAILED to sample ideal within max")
            else:
                print(f"MAX_SAMPLES # sampled in ideal: {len(max_ideal)} (early as {max_ideal[0]}/{args.max_evals}; late as {max_ideal[-1]}/{args.max_evals})")
                # Report rank as +1 to user as they will expect 1-indexing
                print("\t"+f"Best MAX_SAMPLES sample: {1+min(within_max['rank'])} at sample {np.argmin(within_max['rank'])}")

if __name__ == '__main__':
    main()

