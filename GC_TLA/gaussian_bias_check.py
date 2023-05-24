import numpy as np, pandas as pd
import matplotlib.pyplot as plt
#from autotune.space import *
import os, time, argparse
import inspect
from csv import writer
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
from ytopt.search.util import load_from_file


def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="Set seed")
    parser.add_argument('--top', type=float, default=0.3, help="Quantile to train from inputs")
    parser.add_argument('--inputs', type=str, nargs="+", required=True, help="Files for input")
    parser.add_argument('--targets', type=str, nargs='+', required=True,
                        help='problems to use as target tasks')
    parser.add_argument('--model', choices=list(sdv_models.keys()),
                        default='GaussianCopula', help='SDV model')
    parser.add_argument('--no-log-objective', action='store_true',
                        help="Avoid using logarithm on objective values")
    parser.add_argument('--load-log', action='store_true',
                        help="Apply logarithm to loaded TL data")
    parser.add_argument('--speedup', default=None, type=str,
                        help="File to refer to for base speed value")
    parser.add_argument('--ideal-proportion', type=float, default=0.1,
                        help="Ratio of the total population to ideally sample (default: .1)")
    parser.add_argument('--bad-exclude-rate', type=float, default=0.05,
                        help="Proporiton of model space reduction corresponding to ideal population (default: 0.05)")
    parser.add_argument('--success-bar', type=float, default=0.5,
                        help="Confidence threshold to pass for setting budget (default: .5)")
    return parser

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # Ratios must be between 0 and 1 inclusive
    warnings = []
    if args.ideal_proportion < 0:
        warnings.append("Ideal proportion must be >= 0")
    elif args.ideal_proportion > 1:
        warnings.append("Ideal proportion must be <= 1")
    if args.bad_exclude_rate < 0:
        warnings.append("Bad exclude rate must be >= 0")
    elif args.bad_exclude_rate > 1:
        warnings.append("Bad exclude rate must be <= 1")
    if args.success_bar < 0:
        warnings.append("Success bar must be >= 0")
    elif args.success_bar > 1:
        warnings.append("Success bar must be <= 1")
    if len(warnings) > 0:
        raise ValueError("! ".join(warnings)+"!")

    return args

def param_type(k, problem):
    v = problem.problem_params[k]
    if v == 'categorical':
        if hasattr(problem, 'categorical_cast'):
            v = problem.categorical_cast[k]
        else:
            v = 'str'
    if v == 'integer':
        v = 'int64'
    if v == 'str':
        v = 'O'
    return v

def close_enough(frame, rows, column, target, criterion):
    out = []
    # Eliminate rows that are too far from EVER being selected
    if target > criterion[-1]:
        possible = frame[frame[column] > criterion[-1]]
    elif target < criterion[0]:
        possible = frame[frame[column] < criterion[0]]
    else:
        # Find target in the middle using sign change detection in difference
        sign_index = list(np.sign(pd.Series(criterion)-target).diff()[1:].ne(0)).index(True)
        lower, upper = criterion[sign_index:sign_index+2]
        possible = frame[(frame[column] > lower) & (frame[column] < upper)]
    # Prioritize closest rows first
    dists = (possible[column]-target).abs().sort_values().index[:rows]
    return possible.loc[dists].reset_index(drop=True)

def sample_approximate_conditions(sdv_model, model, conditions, criterion, param_names):
    rejections = {}
    duration = {}
    # If model supports conditional sampling, just utilize that
    if conditional_sampling_support[sdv_model]:
        duration['sample'] = time.time()
        samples = model.sample_conditions(conditions)
        duration['sample'] = time.time() - duration['sample']
        return samples, rejections, duration
    # Otherwise, it can be hard to conditionally sample using reject sampling.
    # As such, we do our own reject strategy
    criterion = sorted(criterion)
    requested_rows = sum([_.get_num_rows() for _ in conditions])
    selected = []
    prev_len = -1
    cur_len = 0
    duration['sample'] = 0
    duration['external'] = 0
    # Use lack of change as indication that no additional rows could be found
    while prev_len < requested_rows and cur_len != prev_len:
        prev_len = cur_len
        sample_time = time.time()
        samples = model.sample(num_rows=requested_rows, randomize_samples=False)
        duration['sample'] += time.time() - sample_time
        prune_time = time.time()
        # NO PRUNING
        candidate = samples
        candidate = pd.concat(candidate)#.drop_duplicates(subset=param_names)
        duration['external'] += time.time() - prune_time
        selected.append(candidate)
        cur_len = sum(map(len, selected))
    prune_time = time.time()
    selected = pd.concat(selected)#.drop_duplicates(subset=param_names)
    duration['external'] += time.time() - prune_time
    # FORCE conditions to be held in this data
    for cond in conditions:
        for (col, target) in cond.get_column_values().items():
            selected[col] = target
    return selected, rejections, duration

def csv_to_eval(frame, size):
    csv = frame.drop(columns=['predicted', 'elapsed_sec'])
    csv.insert(loc=len(csv.columns)-1, column='input', value=size)
    csv = csv.rename(columns={'objective': 'runtime'})
    return csv

# Create hypergeometric function
try:
    from math import comb
except ImportError:
    from math import factorial
    def comb(n,k):
        return factorial(n) / (factorial(k) * factorial(n-k))
def hypergeo(i,p,t,k):
    return (comb(i,t)*comb((p-i),(k-t))) / comb(p,k)

def generate_budget(targets, data, inputs, args):
    global time_start

    ORIGINAL_DATA_LENGTH = len(data)
    # All problems (input and target alike) must utilize the same parameters or this is not going to work
    param_names = set(targets[0].params)
    for target_problem in targets[1:]:
        other_names = set(target_problem.params)
        if len(param_names.difference(other_names)) > 0:
            raise ValueError(f"Targets {targets[0].name} and "
                             f"{target_problem.name} utilize different parameters")
    criterion = []
    for input_problem in inputs:
        criterion.append(input_problem.problem_class)
        other_names = set(input_problem.params)
        if len(param_names.difference(other_names)) > 0:
            raise ValueError(f"Target {targets[0].name} and "
                             f"{input_problem.name} utilize different parameters")
    param_names = sorted(param_names)
    n_params = len(param_names)

    # Gather all constraints from all target problems
    constraints = []
    for target in targets:
        constraints.extend(target.constraints)
    if args.model != 'random':
        field_names = ['input']+param_names+['runtime']
        field_transformers = targets[0].problem_params
        model = sdv_models[args.model](field_names=field_names, field_transformers=field_transformers,
                  constraints=constraints, min_value=None, max_value=None)
    else:
        model = None
    # Make conditions for each target
    conditions = []
    for target_problem in targets:
        conditions.append(Condition({'input': target_problem.problem_class},
                                    num_rows=max(100, targets[0].input_space_size)))
    # Initial fit
    if model is not None:
        print(f"Fitting with {len(data)} rows")
        import warnings
        warnings.simplefilter("ignore")
        model.fit(data)
        warnings.simplefilter('default')
    time_start = time.time()
    dup_cols = param_names + ['input']
    durations = []
    generated = []
    eval_master = 0
    time_start = time.time()
    while eval_master < targets[0].input_space_size:
        # Generate prospective points
        warnings.simplefilter("ignore")
        if args.model != 'random':
            # Some SDV models don't realllllly support the kind of conditional sampling we need
            # So this call will bend the condition rules a bit to help them produce usable data
            # until SDV fully supports conditional sampling for those models
            # For any model where SDV has conditional sampling support, this SHOULD utilize SDV's
            # real conditional sampling and bypass the approximation entirely
            ss1, r, d = sample_approximate_conditions(args.model, model, conditions,
                                                      sorted(criterion), param_names)
            for col in param_names:
                ss1[col] = ss1[col].astype(str)
        else:
            # random model is achieved by sampling configurations from the target problem's input space
            columns = ['input']+param_names+['runtime']
            dtypes = [(k,param_type(k, targets[0])) for k in columns]
            random_data = []
            for idx, cond in enumerate(conditions):
                for _ in range(cond.num_rows):
                    # Generate a random valid sample in the parameter space
                    random_params = targets[idx].input_space.sample_configuration().get_dictionary()
                    random_params = [random_params[k] for k in param_names]
                    # Generate the runtime estimate
                    inference = 1.0
                    random_data.append(tuple([cond.column_values['input']]+random_params+[inference]))
            ss1 = np.array(random_data, dtype=dtypes)
            ss1 = pd.DataFrame(ss1, columns=columns)
            for col, dtype in zip(ss1.columns, dtypes):
                if dtype[1] == 'str':
                    ss1[col] = ss1[col].astype('string')
        warnings.simplefilter('default')
        ss = ss1[dup_cols]
        generated.append(ss)
        eval_master += len(ss)
    generated = pd.concat(generated).iloc[:targets[0].input_space_size].drop_duplicates()
    print(f"Generated {len(generated)} unique values after {targets[0].input_space_size} generations ({100*len(generated)/targets[0].input_space_size}%) in {time.time()-time_start} since fit")
    # How many to sample for odds of success?
    Initial_I = int(targets[0].input_space_size * args.ideal_proportion)
    C = int(len(generated))
    reduce_I = int(args.bad_exclude_rate * (targets[0].input_space_size - C))
    I = max(1, Initial_I - reduce_I)
    print(f"Initial Space Size: {targets[0].input_space_size}")
    print(f"Model Generated Space Size: {C} (-{100*(1-(C/targets[0].input_space_size)):.2f}%) after using {100*args.top:.2f}% data totaling {ORIGINAL_DATA_LENGTH} rows")
    print(f"Initial Ideal Size: {Initial_I} based on {100*args.ideal_proportion:.2f}% of initial space")
    print(f"Model Assumed Ideal Size: {I} (-{100*(1-(I/Initial_I)):.2f}%) with {100*args.bad_exclude_rate:.2f}% expected opportunity cost in bias")
    if I > C:
        print(f"Ideal population smaller than biased population!")
        return
    k = 1
    while k < I:
        confidence = sum([hypergeo(I,C,_,k) for _ in range(1,k+1)])
        if confidence >= args.success_bar:
            break
        k += 1
        print(f"Attempt: {k}/{I} | Confidence: {confidence:.4f}",end='\r')
    confidence = sum([hypergeo(I,C,_,k) for _ in range(1,k+1)])
    if k < I or confidence >= args.success_bar:
        print(f"Reached {args.success_bar} probability with {k} samples")
    else:
        print(f"DID NOT REACH CONFIDENCE :: Maximum {100*confidence:.2f}% probability with {k} samples")

def main(args=None):
    args = parse(build(), args)
    print(f"USING {args.model} as model")
    print('how much to train', args.top, 'seed', args.seed, 'ideal proportion', args.ideal_proportion, 'required confidence', args.success_bar,
        'bad exclusion rate', args.bad_exclude_rate)
    # Seed control
    np.random.seed(args.seed)
    # SDV MAY USE TORCH IN LOWER-LEVEL MODULES. CONTROL ITS RNG
    import torch
    torch.manual_seed(args.seed)

    X_opt = []
    # Different objective for speedup
    if args.speedup is not None:
        speed = pd.read_csv(args.speedup)['objective'][0]
    else:
        speed = None

    # Load the input and target problems
    inputs, targets, frames = [], [], []
    # Fetch the target problem(s)'s plopper
    for idx, problemName in enumerate(args.inputs):
        # NOTE: When specified as 'filename.attribute', the second argument 'Problem'
        # is effectively ignored. If only the filename is given (ie: 'filename.py'),
        # defaults to finding the 'Problem' attribute in that file
        if problemName.endswith('.py'):
            attr = 'Problem'
        else:
            pName, attr = problemName.split('.')
            pName += '.py'
        inputs.append(load_from_file(pName, attr))
        # Load the best top x%
        last_in = inputs[-1]
        results_file = last_in.plopper.kernel_dir+"/results_rf_"
        results_file += last_in.dataset_lookup[last_in.problem_class][0].lower()+"_"
        last_in = last_in.__class__.__name__
        results_file += last_in[:last_in.rindex('_')].lower()+".csv"
        if not os.path.exists(results_file):
            # First try backup
            backup_results_file = results_file.rsplit('/',1)
            backup_results_file.insert(1, 'data')
            backup_results_file = "/".join(backup_results_file)
            if not os.path.exists(backup_results_file):
                # Next try replacing '-' with '_'
                dash_results_file = "_".join(results_file.split('-'))
                if not os.path.exists(dash_results_file):
                    dash_backup_results_file = "_".join(backup_results_file.split('-'))
                    if not os.path.exists(dash_backup_results_file):
                        # Execute the input problem and move its results files to the above directory
                        raise ValueError(f"Could not find {results_file} for '{problemName}' "
                                         f"[{inputs[-1].name}] and no backup at {backup_results_file}"
                                         "\nYou may need to run this problem or rename its output "
                                         "as above for the script to locate it")
                    else:
                        print(f"WARNING! {problemName} [{inputs[-1].name}] is using backup data rather "
                                "than original data (Dash-to-Underscore Replacement ON)")
                        results_file = dash_backup_results_file
                else:
                    print("Dash-to-Underscore Replacement ON")
                    results_file = dash_results_file
            else:
                print(f"WARNING! {problemName} [{inputs[-1].name}] is using backup data rather "
                        "than original data")
                results_file = backup_results_file
        dataframe = pd.read_csv(results_file)
        dataframe['input'] = pd.Series(int(inputs[-1].problem_class) for _ in range(len(dataframe.index)))
        dataframe['runtime'] = dataframe['objective']
        if args.load_log:
            dataframe['runtime'] = np.log(dataframe['runtime'])
            if args.speedup is not None:
                speed = np.log(speed)
        q_10_s = np.quantile(dataframe.runtime.values, args.top)
        real_df = dataframe.loc[dataframe['runtime'] <= q_10_s]
        if args.speedup is not None:
            real_df['speedup'] = speed / real_df['runtime']
        real_data = real_df.drop(columns=['elapsed_sec', 'objective'])
        frames.append(real_data)
    # Have to reset the index in case included frames have same index from their original frames
    # Drop the legacy index column as we will not care to recover it and it bothers the shape when
    # recording new evaluations
    real_data = pd.concat(frames).reset_index().drop(columns='index')

    for problemName in args.targets:
        if problemName.endswith('.py'):
            attr = 'Problem'
        else:
            pName, attr = problemName.split('.')
            pName += '.py'
        targets.append(load_from_file(pName, attr))
        # make target evaluations silent as we'll report them on our own
        #targets[-1].silent = True
        # Seed control
        targets[-1].seed(args.seed)
        generate_budget([targets[-1]], real_data, inputs, args)

if __name__ == '__main__':
    main()


