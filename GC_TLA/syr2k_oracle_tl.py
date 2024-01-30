import numpy as np, pandas as pd
import os, time, sys, argparse, pathlib, inspect, warnings
from importlib import import_module
from sdv.single_table import (GaussianCopulaSynthesizer as GaussianCopula,)
#from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE
# Will only use one of these, make a selection dictionary
sdv_models = {'GaussianCopula': GaussianCopula,}
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

from sdv.metadata import SingleTableMetadata
from sdv.sampling.tabular import Condition
from sdv.constraints import ScalarRange
from GC_TLA import gc_tla_utils
# Math for auto-budgeting
try:
    from math import comb
except ImportError:
    from math import factorial
    def comb(n,k):
        return factorial(n) / (factorial(k)*factorial(n-k))
def hypergeo(i,p,t,k):
    return (comb(i,t)*comb((p-i),(k-t))) / comb(p,k)

def build():
    parser = argparse.ArgumentParser()
    experiment = parser.add_argument_group("Experiment")
    experiment.add_argument("--max-evals", type=int, default=30,
                            help="Number of evals (default: %(default)s)")
    experiment.add_argument("--fit-seed", type=int, default=None,
                            help="RNG seed during model fitting (default: %(default)s)")
    experiment.add_argument("--sample-seed", type=int, default=None, nargs="*",
                            help="RNG seed(s) during model inference (default: %(default)s)")
    data = parser.add_argument_group("Data")
    data.add_argument("--top", type=float, default=0.3,
                            help="Filtering quantile (default: %(default)s)")
    data.add_argument("--filter-inverted", action="store_true",
                            help="Invert filtering criterion (default: %(default)s)")
    files = parser.add_argument_group("files")
    files.add_argument("--input-data", nargs="+", required=True,
                            help="Data to concatenate together for model training")
    files.add_argument("--input-attr", nargs="+", required=True,
                            help="Scale indentifiers for each input data (even if pre-identified in data, just make one up)")
    files.add_argument("--problem-file", required=True,
                            help="Path to file that defines importable objects to represent the tuning space and plopping semantics")
    files.add_argument("--transfer-scale", required=True,
                            help="Scale identifier to predict against")
    files.add_argument("--output", default="oracle_gc.csv",
                            help="Path to save results to (if --sample-seed is provided, filename will automatically be noted with each seed value) (default %(default)s)")
    budget = parser.add_argument_group("Budget Estimation")
    budget.add_argument("--skip-auto-budget", action="store_true",
                            help="Always use --max-evals as budet, do not attempt to derive an experiment budget (default: %(default)s)")
    budget.add_argument("--always-max-evals", action="store_true",
                            help="Calculate auto budget, but always utilize max-evals (default: %(default)s)")
    budget.add_argument("--budget-ideal", type=float, default=0.1,
                            help="Ideal Proportion of the population to sample (default %(default)s)")
    budget.add_argument("--budget-attrition", type=float, default=0.05,
                            help="Expected attrition ratio of ideal items removed by GC distribution's specificity (default %(default)s)")
    budget.add_argument("--budget-confidence", type=float, default=0.95,
                            help="Desired confidence value for the budgeted number of trials (default %(default)s)")
    return parser

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    if len(args.input_data) != len(args.input_attr):
        raise ValueError(f"Must supply one --input-attr (observed: {len(args.input_attr)}) per --input-data (observed: {len(args.input_data)})")
    # Seed handling
    if args.fit_seed is not None:
        np.random.seed(args.fit_seed)
        # SDV can use torch in lower-level modules
        import torch
        torch.manual_seed(args.fit_seed)
    if args.sample_seed is None:
        args.sample_seed = [None]
    return args

def import_hook_from_file(fname, attr):
    if type(fname) is not pathlib.Path:
        fname = pathlib.Path(fname)
    dirname = fname.parents[0]
    basename = fname.stem
    sys.path.insert(0, str(dirname))
    module = import_module(basename)
    return getattr(module, attr)

def load_and_filter_input(names, attrs, args):
    concat = []
    for name, attr in zip(names, attrs):
        load = pd.read_csv(name)
        # Inject scale indiator
        indicators = np.asarray(['scale','input','size'])
        indicator = [_ in load.columns for _ in indicators]
        if not any(indicator):
            # Load problem attribute
            problem = import_hook_from_file(args.problem_file, attr)
            load['scale'] = pd.Series(int(problem.problem_class) for _ in range(len(load.index)))
        else:
            # Possibly just rename the data
            if not indicator[0]:
                from_name = indicators[np.where(indicator)[0]][0]
                load = load.rename(columns={from_name: 'scale'})
        # Filter
        if args.filter_inverted:
            qs = np.quantile(load['objective'].values, 1-args.top)
            filtered = load.loc[load['objective'] > qs]
        else:
            qs = np.quantile(load['objective'].values, args.top)
            filtered = load.loc[load['objective'] <= qs]
        filtered = filtered.drop(columns=['elapsed_sec', 'objective'])
        concat.append(filtered)
    combined = pd.concat(concat).reset_index(drop=True)
    # For syr2k and SDV to get along, we need this to be string-typed except the constraint column
    for col in combined.columns:
        if col == 'scale':
            continue
        combined[col] = combined[col].astype(str)
    return combined

def main(args=None):
    args = parse(args)
    # Load inputs
    training_data = load_and_filter_input(args.input_data, args.input_attr, args)
    # Load model evaluator in oracle mode
    evaluator = import_hook_from_file(args.problem_file, f"oracle_{args.transfer_scale}")
    # Construct SDV model
    initial_population = evaluator.input_space_size
    constraints = [{'constraint_class': 'ScalarRange',
                    'constraint_parameters': {
                        'column_name': 'scale',
                        'low_value': min(evaluator.dataset_lookup.keys()),
                        'high_value': max(evaluator.dataset_lookup.keys()),
                        'strict_boundaries': False,
                        },
                   },
                  ]
    # Get budget
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(training_data)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = GaussianCopula(metadata, enforce_min_max_values = False)
        model.add_constraints(constraints=constraints)
        model.fit(training_data)
    if args.skip_auto_budget:
        print(f"Autotuning budget SKIPPED. Using --max-evals={args.max_evals} as budget")
        suggested_budget = args.max_evals
    else:
        budget_conditions = [Condition({'scale': evaluator.problem_class}, num_rows=initial_population)]
        print(f"Determining auto-budget...")
        massive_sample = model.sample_from_conditions(budget_conditions)
        model.reset_sampling() # Ensure budgeting does not change RNG for the observed output (especially if skipped)
        sampled_pop_size = len(massive_sample.drop_duplicates())
        ideal = int(initial_population * args.budget_ideal)
        remaining_ideal = max(1, ideal - int((initial_population - sampled_pop_size) * args.budget_attrition))
        if remaining_ideal > sampled_pop_size:
            print(f"Autotuning budget indeterminate, using --max-evals={args.max_evals} as budget")
            suggested_budget = args.max_evals
        else:
            suggested_budget = 0
            while suggested_budget < sampled_pop_size:
                suggested_budget += 1
                confidence = sum([hypergeo(remaining_ideal, sampled_pop_size, _, suggested_budget) for _ in range(1, suggested_budget+1)])
                if confidence >= args.budget_confidence:
                    break
            if confidence >= args.budget_confidence:
                print(f"Autotuning budget {suggested_budget} attains accepted confidence: {confidence:.4f} >= {args.budget_confidence}")
            else:
                print(f"Autotuning budget {suggested_budget} fails to meet desired confidence {args.budget_confidence}. Best confidence: {confidence:.4f}")
            if suggested_budget > args.max_evals:
                print(f"Reducing budget {suggested_budget} to --max-evals value: {args.max_evals}")
                suggested_budget = args.max_evals
            elif args.always_max_evals:
                print(f"Overriding budget to --max-evals value ({args.max_evals}) due to --always-max-evals")
                suggested_budget = args.max_evals
    # Actual runtime experiment (using oracle)
    eval_conditions = [Condition({'scale': evaluator.problem_class}, num_rows=suggested_budget)]
    for sample_seed in args.sample_seed:
        if sample_seed is not None:
            # ---- UGLY WORKAROUND -----
            # SDV forces a fixed RNG seed every time you sample with conditions :(
            #
            # The public API only provides .reset_sampling() to alter RNG, which returns RNG to the post-fitting state, which is ... insufficient.
            # This only allows you to repeat the same RNG for different conditions, not to repeat RNG or alter RNG.
            # As you can see, we use .reset_sampling() above for good reason, but this will not permit us all of the RNG control necessary to interact
            # with the learned distribution in statistically analyzable manners.
            # We need actually different RNG with the same condition / data for variance analysis (and many other use cases),
            # but there currently isn't an "accepted" API to achieve this.
            #
            # As such, the next line of code is VERY unstable and SDV may deprecate or break it at any time without warning.
            # As is often the case, SDV fixed this in the past and un-fixed it later, so this can be considered evolving behavior
            # in other words: behavior is unpredictable based on your SDV installation version. Caution is advised.
            # The current line of code in this file uses the user-level SDV object as that should remain SLIGHTLY more stable,
            # but the intended end effector is buried in the secondary sub-model which can be represented most directly as:
            #
            # model._model.random_state.seed(sample_seed)
            # ----- END DISCUSSION ------
            model._set_random_state(sample_seed)
            # Customize output name
            out_name = pathlib.Path(args.output)
            out_name = out_name.parents[0].joinpath(out_name.stem+f"_seed_{sample_seed}"+out_name.suffix)
        else:
            out_name = args.output
        conditional_samples = model.sample_from_conditions(eval_conditions)
        results = pd.DataFrame([], columns=training_data.columns)
        for idx, row in conditional_samples.iterrows():
            row_params = [_ for _ in row[:-1]]
            objective = evaluator.oracle_search(row_params)
            row_data = dict((c,d) for (c,d) in row.items())
            row_data['objective'] = objective
            results = pd.concat((results,pd.DataFrame(row_data, index=[0]))).reset_index(drop=True)
        results.to_csv(out_name, index=False)

if __name__ == '__main__':
    main()

