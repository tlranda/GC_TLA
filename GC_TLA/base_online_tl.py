import numpy as np, pandas as pd
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
    parser.add_argument('--max-evals', type=int, default=10,
                        help='maximum number of evaluations')
    parser.add_argument('--n-refit', type=int, default=0,
                        help='refit the model')
    parser.add_argument('--seed', type=int, default=1234,
                        help='set seed')
    parser.add_argument('--top', type=float, default=0.1,
                        help='how much to train')
    parser.add_argument('--fit-bottom', action='store_true',
                        help='invert the trimming criterion')
    parser.add_argument('--inputs', type=str, nargs='+', required=True,
                        help='problems to use as input')
    parser.add_argument('--targets', type=str, nargs='+', required=True,
                        help='problems to use as target tasks')
    parser.add_argument('--model', choices=list(sdv_models.keys()),
                        default='GaussianCopula', help='SDV model')
    parser.add_argument('--single-target', action='store_true',
                        help='Treat each target as a unique problem (default: solve all targets at once)')
    parser.add_argument('--exhaust', action='store_true',
                        help="Exhaust all configurations in the target problem (default: don't do this)")
    parser.add_argument('--output-prefix', type=str, default='results_sdv',
                        help='Output files are created using this prefix (default: [results_sdv]*.csv)')
    parser.add_argument('--no-log-objective', action='store_true',
                        help="Avoid using logarithm on objective values")
    parser.add_argument('--load-log', action='store_true',
                        help="Apply logarithm to loaded TL data")
    parser.add_argument('--speedup', default=None, type=str,
                        help="File to refer to for base speed value")
    parser.add_argument('--resume', default=None, type=str,
                        help="Previous run to resume from (if specified)")
    parser.add_argument('--resume-fit', default=0, type=int,
                        help="Rows to refit on from the resuming data (default 0 -- blank model used, use -1 to infer the last fit)")
    parser.add_argument('--all-file', default=None, type=str,
                        help="Look up evaluation results from this file instead of actually evaluating via objective")
    parser.add_argument('--skip-evals', action='store_true',
                        help="Prevent actual objective calculation")
    return parser

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.resume is not None:
        try:
            x = open(args.resume, 'r')
        except IOError:
            # Nonexistent resume files should fail gracefully with a warning
            print(f"! warning! could not resume from {args.resume}: File Not Found")
            args.resume = None
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
        candidate = []
        for cond in conditions:
            n_rows = cond.get_num_rows()
            for (col, target) in cond.get_column_values().items():
                candidate.append(close_enough(samples, n_rows, col, target, criterion))
                # Difference of length between samples::candidate[-1] == closeness trimming
                if 'closeness' in rejections.keys():
                    rejections['closeness'] += len(samples)-len(candidate[-1])
                else:
                    rejections['closeness'] = len(samples)-len(candidate[-1])
        # Difference here represents redundancy in sampling == duplicate trimming
        if 'dup_samples' in rejections.keys():
            rejections['dup_samples'] += sum([len(_) for _ in candidate]) # Prepare difference
        else:
            rejections['dup_samples'] = sum([len(_) for _ in candidate])
        candidate = pd.concat(candidate).drop_duplicates(subset=param_names)
        duration['external'] += time.time() - prune_time
        rejections['dup_samples'] -= len(candidate) # Correct difference magnitude after drop-dup_samples
        selected.append(candidate)
        cur_len = sum(map(len, selected))
    # Difference here represents ADDITIONAL redundancy in sampling == duplicate trimming pt 2
    if 'dup_batch' in rejections.keys():
        rejections['dup_batch'] += sum([len(_) for _ in selected]) # Prepare difference
    else:
        rejections['dup_batch'] = sum([len(_) for _ in selected])
    prune_time = time.time()
    selected = pd.concat(selected).drop_duplicates(subset=param_names)
    duration['external'] += time.time() - prune_time
    rejections['dup_batch'] -= len(selected) # Correct difference magnitude after drop-duplicates
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

def exhaust(target, data, inputs, args, fname, speed = None):
    # Rather than using a model, iterate ALL configurations
    global time_start

    # All problems (input and target alike) must utilize the same parameters or this is not going to work
    param_names = set(target.params)
    params = []
    n_configs = 1
    for _ in target.input_space.get_hyperparameters():
        try:
            params.append(_.sequence)
            n_configs *= len(_.sequence)
        except AttributeError:
            params.append(_.choices)
            n_configs *= len(_.choices)
    args.max_evals = n_configs
    criterion = []
    for input_problem in inputs:
        criterion.append(input_problem.problem_class)
        other_names = set(input_problem.params)
        if len(param_names.difference(other_names)) > 0:
            raise ValueError(f"Target {target.name} and "
                             f"{input_problem.name} utilize different parameters")
    param_names = sorted(param_names)
    n_params = len(param_names)

    # Gather all constraints from all target problems
    csv_fields = param_names+['objective','predicted','elapsed_sec']

    # Load resumable data and preserve it in new run so no need to re-merge new and existing data
    if args.resume is not None:
        try:
            # Better to recompute a bad line than die entirely -- use warning to alert to issue
            evals_infer = pd.read_csv(args.resume)
        except IOError:
            print(f"WARNING: Could not resume {args.resume}")
            evals_infer = None
    else:
        evals_infer = None

    # writing to csv file
    with open(fname, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = writer(csvfile)
        # writing the fields
        csvwriter.writerow(csv_fields)

        # Load resumable data and preserve it in new run so no need to re-merge new and existing data
        if evals_infer is not None:
            for ss in evals_infer.iterrows():
                csvwriter.writerow(ss[1].values)
            csvfile.flush()
            loaded_evals = evals_infer
            evals_infer = csv_to_eval(evals_infer, target.problem_class)
            if args.resume_fit < 0 and args.n_refit > 0:
                # Infer best resume point based on refit
                loaded = len(evals_infer)
                args.resume_fit = loaded - (loaded % args.n_refit)
            if args.resume_fit > 0:
                data = pd.concat((data, evals_infer[:args.resume_fit])).reset_index(drop=True)
        else:
            evals_infer = []

        # Make conditions for each target
        eval_master = len(evals_infer)
        if args.resume is not None:
            # There may be additional data AFTER this fit to keep track of
            if args.resume_fit != len(evals_infer):
                data = pd.concat((data, evals_infer[args.resume_fit:])).reset_index(drop=True)
            # Convert evals_infer to a list now (only needs the runtimes)
            evals_infer = evals_infer['runtime'].to_list()
            resume_utilized = False
        else:
            resume_utilized = True
        time_start = time.time()
        import itertools
        for i, config in enumerate(itertools.product(*params)):
            # Resume, skip this much
            #if i < eval_master:
            #    continue
            sample_point = dict((pp,vv) for (pp,vv) in zip(param_names, config))
            for j in [f'p{x}' for x in range(3,6)]:
                sample_point[j] = int(sample_point[j])
            look = tuple([sample_point[_] for _ in param_names])
            n_matching = np.where((loaded_evals[param_names] == look).sum(1) == 6)[0]
            if len(n_matching) > 0:
                if len(n_matching) > 1:
                    print(f"WARN: Found duplicate: {sample_point} @{i}")
                continue
            print(f"Eval {i}/{args.max_evals}")
            print(sample_point)
            if speed is None:
                evals_infer.append(target.objective(sample_point))
            else:
                evals_infer.append(speed / target.objective(sample_point))
            now = time.time()
            elapsed = now - time_start
            ss = [_ for _ in config]
            ss.extend([evals_infer[-1], target.problem_class, elapsed])
            csvwriter.writerow(ss)
            csvfile.flush()
    csvfile.close()

def online(targets, data, inputs, args, fname, speed = None, exhaust = None):
    global time_start

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
    csv_fields = param_names+['objective','predicted','elapsed_sec']
    for i in range(len(targets)-1):
        csv_fields.extend([f'objective_{i}',f'predicted_{i}',f'elapsed_sec_{i}'])

    # Load resumable data and preserve it in new run so no need to re-merge new and existing data
    if args.resume is not None:
        try:
            evals_infer = pd.read_csv(args.resume)
        except IOError:
            print(f"WARNING: Could not resume {args.resume}")
            evals_infer = None
    else:
        evals_infer = None

    # Possibly substitue REAL evaluations with lookups
    if exhaust is not None:
        if type(exhaust) is str:
            exhaust = pd.read_csv(exhaust).sort_values(by='objective').reset_index(drop=True)

    # writing to csv file
    supplementary_fname = fname[:-4]
    if supplementary_fname.endswith('_ALL'):
        supplementary_fname = supplementary_fname[:-4]
    supplementary_fname += '_trace.csv'
    with open(supplementary_fname, 'w') as suppfile:
        suppwriter = writer(suppfile)
        suppwriter.writerow(['trial','generate', 'reject','close','sample','batch','prior','sample','external'])
        trial = 0
        with open(fname, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = writer(csvfile)
            # writing the fields
            csvwriter.writerow(csv_fields)

            # Load resumable data and preserve it in new run so no need to re-merge new and existing data
            if evals_infer is not None:
                for ss in evals_infer.iterrows():
                    csvwriter.writerow(ss[1].values)
                csvfile.flush()
                evals_infer = csv_to_eval(evals_infer, targets[0].problem_class)
                if args.resume_fit < 0 and args.n_refit > 0:
                    # Infer best resume point based on refit
                    loaded = len(evals_infer)
                    args.resume_fit = loaded - (loaded % args.n_refit)
                if args.resume_fit > 0:
                    data = pd.concat((data, evals_infer[:args.resume_fit])).reset_index(drop=True)
            else:
                evals_infer = []

            # Make conditions for each target
            conditions = []
            for target_problem in targets:
                conditions.append(Condition({'input': target_problem.problem_class},
                                            num_rows=max(100, args.max_evals)))
            eval_master = len(evals_infer)
            # Initial fit
            if model is not None:
                print(f"Fitting with {len(data)} rows")
                import warnings
                warnings.simplefilter("ignore")
                model.fit(data)
                warnings.simplefilter('default')
            if args.resume is not None:
                # There may be additional data AFTER this fit to keep track of
                if args.resume_fit != len(evals_infer):
                    data = pd.concat((data, evals_infer[args.resume_fit:])).reset_index(drop=True)
                # Convert evals_infer to a list now (only needs the runtimes)
                evals_infer = evals_infer['runtime'].to_list()
                resume_utilized = False
            else:
                resume_utilized = True
            time_start = time.time()
            EFFICIENCY = {'generated': 0, 'rejections': {}, 'durations': {}}
            dup_cols = param_names + ['input']
            while eval_master < args.max_evals:
                rejections = {'closeness': 0, 'dup_samples': 0, 'dup_batch': 0, 'prior_consideration': 0}
                durations = {'sample': 0, 'external': 0}
                # Generate prospective points
                if args.model != 'random':
                    # Some SDV models don't realllllly support the kind of conditional sampling we need
                    # So this call will bend the condition rules a bit to help them produce usable data
                    # until SDV fully supports conditional sampling for those models
                    # For any model where SDV has conditional sampling support, this SHOULD utilize SDV's
                    # real conditional sampling and bypass the approximation entirely
                    ss1, r, d = sample_approximate_conditions(args.model, model, conditions,
                                                              sorted(criterion), param_names)
                    rejections.update(r)
                    durations.update(d)
                    for col in param_names:
                        ss1[col] = ss1[col].astype(str)
                else:
                    # random model is achieved by sampling configurations from the target problem's input space
                    columns = ['input']+param_names+['runtime']
                    dtypes = [(k,param_type(k, targets[0])) for k in columns]
                    random_data = []
                    durations['sample'] = time.time()
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
                    durations['sample'] = time.time() - durations['sample']
                    for col, dtype in zip(ss1.columns, dtypes):
                        if dtype[1] == 'str':
                            ss1[col] = ss1[col].astype('string')
                # Don't evaluate the exact same parameter configuration multiple times in a fitting round
                rejections['prior_consideration'] = len(ss1) # Pre-add FULL set to see what survives this round of dropping
                ss1 = ss1.drop_duplicates(subset=param_names, keep="first")
                # Drop generated values that have already been evaluated
                # STACK known / evaluated values with new predictions (include runtime columns but we won't compare on them)
                stacked = pd.concat([data, ss1])
                # Switch idx = when we return to ss1 data, ie after len(data) since we just got a new index for stacked
                switch = len(data)
                # Include the stacked values (after switch) that are NOT duplicates
                ss1 = stacked.iloc[switch:][~stacked.duplicated(subset=dup_cols).iloc[switch:].values]
                rejections['prior_consideration'] -= len(ss1) # Subtract the ones that remained new
                EFFICIENCY['generated'] += len(ss1) # ACTUALLY USEFUL RUNNING COUNT
                for key, value in rejections.items():
                    if key not in EFFICIENCY['rejections'].keys():
                        EFFICIENCY['rejections'][key] = value
                    else:
                        EFFICIENCY['rejections'][key] += value
                for key, value in durations.items():
                    if key not in EFFICIENCY['durations'].keys():
                        EFFICIENCY['durations'][key] = value
                    else:
                        EFFICIENCY['durations'][key] += value
                ss = ss1.sort_values(by='runtime')#, ascending=False)
                new_sdv = ss[:args.max_evals]
                eval_update = 0 if resume_utilized else len(evals_infer)-args.resume_fit
                if not resume_utilized:
                    resume_utilized = True
                stop = False
                while not stop:
                    for row in new_sdv.iterrows():
                        sample_point_val = row[1][param_names+['input','runtime']]
                        sample_point = dict((pp,vv) for (pp,vv) in zip(param_names, sample_point_val))
                        ss = []
                        for target_problem in targets:
                            # Use the target problem's .objective() call to generate an evaluation
                            if not args.skip_evals:
                                print(f"Eval {eval_master+1}/{args.max_evals}")
                            # Determine objective and search_equals
                            if exhaust is None:
                                search_equals = tuple(row[1][param_names+['input']].tolist())
                                if not args.skip_evals:
                                    objective = target_problem.objective(sample_point)
                                else:
                                    objective = 1
                            else:
                                # Use lookup in exhaust to find the objective!
                                search_equals = tuple(row[1][param_names].tolist())
                                n_matching_columns = (exhaust[param_names].astype(str) == search_equals).sum(1)
                                full_match_idx = np.where(n_matching_columns == n_params)[0]
                                if len(full_match_idx) == 0:
                                    raise ValueError(f"Failed to find tuple {list(search_equals)} in '--all-file' data")
                                objective = exhaust.iloc[full_match_idx]['objective'].values[0]
                                # Add problem size back into search_equals
                                search_equals = tuple(list(search_equals)+[row[1]['input']])
                                print(f"All file rank: {full_match_idx[0]} / {len(exhaust)}")
                            if speed is None:
                                evals_infer.append(objective)
                            else:
                                evals_infer.append(speed / objective)
                            #print(target_problem.name, sample_point, evals_infer[-1])
                            now = time.time()
                            elapsed = now - time_start
                            if ss == []:
                                ss = [sample_point[k] for k in param_names]
                                ss += [evals_infer[-1]]+[sample_point_val[-1]]
                                ss += [elapsed]
                            else:
                                # Append new target data to the CSV row
                                ss2 = [evals_infer[-1]]+[sample_point_val[-1]]
                                ss2 += [elapsed]
                                ss.extend(ss2)
                            # Data tracks problem parameters + time (perhaps log time instead)
                            evaluated = list(search_equals)
                            # Insert runtime before the problem class size
                            if args.no_log_objective:
                                evaluated.append(float(evals_infer[-1]))
                            else:
                                evaluated.append(float(np.log(evals_infer[-1])))
                            # Add record into dataset
                            data.loc[max(data.index)+1] = evaluated
                        # Record in CSV and update iteration
                        # CSV records problem parameters, time (perhaps log), predicted time (perhaps predicting log), elapsed time
                        csvwriter.writerow(ss)
                        csvfile.flush()
                        eval_update += 1
                        eval_master += 1
                        # Quit when out of evaluations (no final refit check needed)
                        if eval_master >= args.max_evals:
                            stop = True
                            break
                        # Important to run refit AFTER at least one evaluation is done, else we
                        # infinite loop when args.n_refit=0 (never)
                        if eval_update == args.n_refit:
                            # update model
                            if model is not None:
                                print("REFIT")
                                warnings.simplefilter("ignore")
                                model.fit(data)
                                warnings.simplefilter('default')
                            stop = True
                            break
                    if args.n_refit == 0:
                        stop = True
                # AFTER exhausting the batch, show the generative statistics
                suppwriter.writerow([trial, EFFICIENCY['generated'], sum(EFFICIENCY['rejections'].values())]+
                                     list(EFFICIENCY['rejections'].values())+list(EFFICIENCY['durations'].values()))
                trial += 1
                print(f"Generate: {EFFICIENCY['generated']} | ",end='')
                print(f"REJECT: {sum(EFFICIENCY['rejections'].values())} | ",end='')
                print(f"RATIO: {EFFICIENCY['generated']/sum(EFFICIENCY['rejections'].values())}")
                print(EFFICIENCY['rejections'])
                print(EFFICIENCY['durations'])
        csvfile.close()

def load_input(obj, problemName, speed, args):
    if obj.use_oracle:
        fname = obj.plopper.kernel_dir+"/oracle_bo_"
    else:
        fname = obj.plopper.kernel_dir+"/results_rf_"
    fname += obj.dataset_lookup[obj.problem_class][0].lower()+"_"
    clsname = obj.__class__.__name__
    fname += clsname[:clsname.rindex('_')].lower()+".csv"
    if not os.path.exists(fname):
        # First try backup
        backup_fname = fname.rsplit('/',1)
        backup_fname.insert(1, 'data')
        backup_fname = "/".join(backup_fname)
        if not os.path.exists(backup_fname):
            # Next try replacing '-' with '_'
            dash_fname = "_".join(fname.split('-'))
            if not os.path.exists(dash_fname):
                dash_backup_fname = "_".join(backup_fname.split('-'))
                if not os.path.exists(dash_backup_fname):
                    # Execute the input problem and move its results files to the above directory
                    raise ValueError(f"Could not find {fname} for '{problemName}' "
                                     f"[{obj.name}] and no backup at {backup_fname}"
                                     "\nYou may need to run this problem or rename its output "
                                     "as above for the script to locate it")
                else:
                    print(f"WARNING! {problemName} [{obj.name}] is using backup data rather "
                            "than original data (Dash-to-Underscore Replacement ON)")
                    fname = dash_backup_fname
            else:
                print("Dash-to-Underscore Replacement ON")
                fname = dash_fname
        else:
            print(f"WARNING! {problemName} [{obj.name}] is using backup data rather "
                    "than original data")
            fname = backup_fname
    dataframe = pd.read_csv(fname)
    dataframe['input'] = pd.Series(int(obj.problem_class) for _ in range(len(dataframe.index)))
    dataframe['runtime'] = dataframe['objective']
    if args.load_log:
        dataframe['runtime'] = np.log(dataframe['runtime'])
        if args.speedup is not None:
            speed = np.log(speed)
    if args.fit_bottom:
        q_10_s = np.quantile(dataframe.runtime.values, 1-args.top)
        real_df = dataframe.loc[dataframe['runtime'] > q_10_s]
    else:
        q_10_s = np.quantile(dataframe.runtime.values, args.top)
        real_df = dataframe.loc[dataframe['runtime'] <= q_10_s]
    if args.speedup is not None:
        real_df['speedup'] = speed / real_df['runtime']
    real_data = real_df.drop(columns=['elapsed_sec', 'objective'])
    return real_data

def main(args=None):
    args = parse(build(), args)
    output_prefix = args.output_prefix
    print(f"USING {args.model} for constraints")
    print('max_evals', args.max_evals, 'number of refit', args.n_refit, 'how much to train', args.top,
          'seed', args.seed)
    # Seed control
    np.random.seed(args.seed)
    # SDV MAY USE TORCH IN LOWER-LEVEL MODULES. CONTROL ITS RNG
    import torch
    torch.manual_seed(args.seed)

    X_opt = []
    print ('----------------------------- how much data to use?', args.top)

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
        frames.append(load_input(inputs[-1], problemName, speed, args))
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
        # Single-target mode
        if args.single_target:
            online([targets[-1]], real_data, inputs, args, f"{output_prefix}_{targets[-1].name}.csv", speed)
    if args.exhaust:
        exhaust(targets[-1], real_data, inputs, args, f"{output_prefix}_{targets[-1].name}_EXHAUST.csv", speed)
    elif not args.single_target:
        online(targets, real_data, inputs, args, f"{output_prefix}_ALL.csv", speed, args.all_file)

if __name__ == '__main__':
    main()

