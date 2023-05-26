import numpy as np, pandas as pd, os, argparse, matplotlib
# Change backend if need be
# matplotlib.use_backend()
font = {'size': 14,
        'family': 'serif',
        }
lines = {'linewidth': 3,
         'markersize': 6,
        }
matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
rcparams = {'axes.labelsize': 14,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            }
plt.rcParams.update(rcparams)
import matplotlib.colors as mcolors
# Get legend names from matplotlib
from matplotlib.offsetbox import AnchoredOffsetbox
legend_codes = list(AnchoredOffsetbox.codes.keys())+['best']

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--output", type=str, default="fig", help="Prefix for output images")
    prs.add_argument("--inputs", type=str, nargs="+", help="Files to read for plots")
    prs.add_argument("--bests", type=str, nargs="*", help="Traces to treat as best-so-far")
    prs.add_argument("--baseline-best", type=str, nargs="*", help="Traces to treat as BEST of best so far")
    prs.add_argument("--xfers", type=str, nargs="*", help="Traces to treat as xfers (ONE PLOT PER FILE)")
    prs.add_argument("--pca", type=str, nargs="*", help="Plot as PCA (don't mix with other plots plz)")
    prs.add_argument("--pca-problem", type=str, default="", help="Problem.Attr notation to load space from (must be module or CWD/* to function)")
    prs.add_argument("--pca-points", type=int, default=None, help="Limit the number of points used for PCA (spread by quantiles, default ALL points used)")
    prs.add_argument("--pca-tops", type=float, nargs='*', help="Top%% to use for each PCA file (disables point-count/quantization, keeps k-ranking)")
    prs.add_argument("--pca-algorithm", choices=['pca', 'tsne'], default='tsne', help="Algorithm to use for dimensionality reduction (default tsne)")
    prs.add_argument("--as-speedup-vs", type=str, default=None, help="Convert objectives to speedup compared against this value (float or CSV filename)")
    prs.add_argument("--show", action="store_true", help="Show figures rather than save to file")
    prs.add_argument("--legend", choices=legend_codes, nargs="*", default=None, help="Legend location (default none). Two-word legends should be quoted on command line")
    prs.add_argument("--budget", type=int, default=None, help="Indicate performance of each technique's best result at a budgeted # of evaluations")
    prs.add_argument("--tzero", action="store_true", help="Assume a t=0 / step=0 entry for each loaded source where the objective == args.tobjective")
    prs.add_argument("--tobjective", type=float, default=1.0, help="Default t=0 / step = 0 objective (default: 1.0)")
    prs.add_argument("--minmax", action="store_true", help="Include min and max lines")
    prs.add_argument("--stddev", action="store_true", help="Include stddev range area")
    prs.add_argument("--current", action="store_true", help="Include area for actual evaluation")
    prs.add_argument("--x-axis", choices=["evaluation", "walltime"], default="evaluation", help="Unit for x-axis")
    prs.add_argument("--log-x", action="store_true", help="Logarithmic x axis")
    prs.add_argument("--log-y", action="store_true", help="Logarithmic y axis")
    prs.add_argument("--below-zero", action="store_true", help="Allow plotted values to be <0")
    prs.add_argument("--unname-prefix", type=str, default="", help="Prefix from filenames to remove from line labels")
    prs.add_argument("--drop-extension", action="store_true", help="Remove file extension from name")
    prs.add_argument("--trim", type=str, nargs="*", help="Trim these files to where the objective changes")
    prs.add_argument("--fig-dims", metavar=("Xinches", "Yinches"), nargs=2, type=float,
                     default=plt.rcParams["figure.figsize"], help="Figure size in inches "
                     f"(default is {plt.rcParams['figure.figsize']})")
    prs.add_argument("--fig-pts", type=float, default=None, help="Specify figure size using LaTeX points and Golden Ratio")
    prs.add_argument("--format", choices=["png", "pdf", "svg","jpeg"], default="pdf", help="Format to save outputs in")
    prs.add_argument("--synchronous", action="store_true", help="Synchronize mean time across seeds for wall-time plots")
    prs.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    prs.add_argument("--no-text", action="store_true", help="Skip text generation")
    prs.add_argument("--merge-dirs", action="store_true", help="Ignore directories when combining files")
    prs.add_argument("--top", type=float, default=None, help="Change to plot where y increments by 1 each time a new evaluation is turned in that is at or above this percentile of performance (1 == best, 0 == worst)")
    prs.add_argument("--global-top", action="store_true", help="Use a single top value across ALL loaded data")
    prs.add_argument("--max-objective", action="store_true", help="Objective is MAXIMIZE not MINIMIZE (default MINIMIZE)")
    prs.add_argument("--ignore", type=str, nargs="*", help="Files to unglob")
    prs.add_argument("--drop-seeds", type=int, nargs="*", help="Seeds to remove (in ascending 1-based rank order by performance, can use negative numbers for nth best)")
    prs.add_argument("--cutoff", action="store_true", help="Halt plotting points after the maximum is achieved")
    prs.add_argument("--drop-overhead", action="store_true", help="Attempt to remove initialization overhead time in seconds")
    prs.add_argument("--clean-names", action="store_true", help="Use a cleaner name format to label lines (better for final figures)")
    return prs

def set_size(width, fraction=1, subplots=(1,1)):
    # SOURCE:
    # https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    # Set figure dimensions to avoid scaling in LaTeX
    # Get your width from the log file of your compiled file using "\showthe\textwdith" or "\showthe\columnwidth"
    # You can grep/search it for that command and the line above will have the value
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    print(f"Calculate {width} to represent inches: {fig_width_in} by {fig_height_in}")
    return (fig_width_in, fig_height_in)

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.stddev and args.current:
        raise ValueError("Only ONE of --stddev and --current can be used in the same plot")
    if args.trim is None:
        args.trim = list()
    if not args.max_objective:
        # Quantile should be (1 - %) if MINIMIZE (lower is better)
        if args.top is not None:
            args.top = 1 - args.top
        if args.pca_tops is not None:
            args.pca_tops = [1-q for q in args.pca_tops]
    # Go through plottable lists and remove things that were supposed to be unglobbed
    if args.ignore is not None:
        plot_globs = ['inputs', 'bests', 'baseline_best', 'xfers', 'pca']
        for glob in plot_globs:
            attr = getattr(args, glob)
            if attr is not None:
                allowed = []
                for fname in attr:
                    if fname not in args.ignore:
                        allowed.append(fname)
                setattr(args, glob, allowed)
    if args.pca is not None and args.pca != [] and args.pca_problem == "":
        raise ValueError("Must define a pca problem along with PCA plots (--pca-problem)")
    if args.pca_tops is not None and args.pca_tops != [] and len(args.pca_tops) != len(args.pca):
        raise ValueError(f"When specified, --pca-tops (length {len(args.pca_tops)}) must have one entry per PCA type input ({len(args.pca)})")
    if args.drop_seeds == []:
        args.drop_seeds = None
    if args.as_speedup_vs is not None:
        try:
            args.as_speedup_vs = float(args.as_speedup_vs)
        except ValueError:
            args.as_speedup_vs = pd.read_csv(args.as_speedup_vs).iloc[0]['objective']
    if args.fig_pts is not None:
        args.fig_dims = set_size(args.fig_pts)
    return args

substitute = {'BOOTSTRAP': "Bootstrap",
              'NO': 'Gaussian Copula',
              'gptune': "GPTune",
              'GPTune': "GPTune",
              'REFIT': "Gaussian Copula with Refit",
              'results': "BO",
              'bo': 'BO',
              'gc': 'Gaussian Copula'}
HAS_NOT_WARNED_MAKE_SEED_INVARIANT_NAME = True

benchmark_names = {'Lu': 'LU',
                    'Amg': 'AMG',
                    'Rsbench': 'RSBench',
                    'Xsbench': 'XSBench',
                    'Sw4lite': 'SW4Lite',
                    }
def try_familiar(name):
    if name in benchmark_names.keys():
        return benchmark_names[name]
    return name

def make_seed_invariant_name(name, args):
    directory = os.path.dirname(name) if not args.merge_dirs else 'MERGE'
    name = os.path.basename(name)
    name_dot, ext = name.rsplit('.',1)
    if name_dot.endswith("_ALL"):
        name_dot = name_dot[:-4]
    try:
        base, seed = name_dot.rsplit('_',1)
        intval = int(seed)
        name = base
    except ValueError:
        if '.' in name and args.drop_extension:
            name, _ = name.rsplit('.',1)
        name = name.lstrip("_")
    else:
        if args.unname_prefix != "" and name.startswith(args.unname_prefix):
            name = name[len(args.unname_prefix):]
        if '.' in name and args.drop_extension:
            name, _ = name.rsplit('.',1)
    name = name.lstrip("_")
    suggest_legend_title = None
    if args.clean_names:
        # Attempt to identify the number of _ characters from the directory name
        if directory.endswith('_exp'):
            temp_split = directory.split('/')
            decide = lambda string : True if '_exp' in string else False
            has_exp = temp_split[[decide(_) for _ in temp_split].index(True)]
            # Subtract 1 due to _exp being a split
            suggest_benchmark_length = len(has_exp.lstrip('_').split('_'))-1
        else:
            temp_split = os.path.dirname(os.path.abspath(name)).split('/')
            decide = lambda string : True if '_exp' in string else False
            try:
                has_exp = temp_split[[decide(_) for _ in temp_split].index(True)]
                # Subtract 1 due to _exp being a split
                suggest_benchmark_length = len(has_exp.lstrip('_').split('_'))-1
                #global HAS_NOT_WARNED_MAKE_SEED_INVARIANT_NAME
                #if HAS_NOT_WARNED_MAKE_SEED_INVARIANT_NAME:
                #    print("WARNING: Unable to determine benchmark name length--this may cause errors.")
                #    print("You can address this by ensuring data is encapsulated in a directory visible on relative paths with the name \"{benchmark_name}_exp\"")
                #    HAS_NOT_WARNED_MAKE_SEED_INVARIANT_NAME = False
                #suggest_benchmark_length = 1
            except:
                suggest_benchmark_length = 1
        name_split = name.split('_')
        # Decompose for ease of semantics
        if name.startswith('results'):
            if 'gptune' in name.lower() and len(name_split) < 4:
                off = 0
            else:
                off = suggest_benchmark_length
            name_split = {'benchmark': '_'.join([_.capitalize() for _ in name_split[-off:]]).rstrip('.csv'),
                          'size': name_split[-1-off].upper(),
                          'short_identifier': substitute[name_split[0]] if 'gptune' not in name.lower() else 'GPTune',
                          'full_identifier': '_'.join(name_split[:-1-off])}
        elif 'xfer' in name:
            name_split = {'benchmark': name[len('xfer_results_')+1:].capitalize(),
                          'size': 'Force Transfer'}
            name_split['short_identifier'] = f"XFER {name_split['benchmark']}"
            name_split['full_identifier'] = f"Force Transfer {name_split['benchmark']}"
        else:
            name_split = {'benchmark': '_'.join([_.capitalize() for _ in name_split[:suggest_benchmark_length]]),
                          'size': name_split[-1],
                          'short_identifier': substitute[name_split[suggest_benchmark_length]],
                          'full_identifier': '_'.join(name_split[suggest_benchmark_length+1:-1])}
        # Reorder in reconstruction
        name = name_split['short_identifier']
        suggest_legend_title = f"{name_split['size']} {try_familiar(name_split['benchmark'].replace('_', ' '))}"
    return name, directory, suggest_legend_title

def make_seed_invariant_name(name, args):
    fullsplit = name.split('/')
    directory = fullsplit[1]
    for key in substitute.keys():
        if key.lower() in directory.lower():
            name = substitute[key]
            break
    namesplit = fullsplit[-1].split('_')
    bench = namesplit[0]
    for key, value in benchmark_names.items():
        if key.lower() == bench.lower():
            bench = value
            break
    size = namesplit[1]
    seed = namesplit[2]
    suggest_legend_title = f"{size} {bench}"
    return name, directory, suggest_legend_title

def make_baseline_name(name, args, df, col):
    name, directory, _ = make_seed_invariant_name(name, args)
    if args.max_objective:
        return name + f"_using_eval_{df[col].idxmax()+1}/{max(df[col].index)+1}", directory
    else:
        return name + f"_using_eval_{df[col].idxmin()+1}/{max(df[col].index)+1}", directory

def drop_seeds(data, args):
    if args.drop_seeds is None:
        return data
    for entry in data:
        # Fix relative indices
        drop_seeds = []
        new_data = []
        for rank in args.drop_seeds:
            if rank < 0:
                # Subtract 1-based index from +1'd length
                drop_seeds.append(len(entry)+rank)
            else:
                # Subtract 1-based index
                drop_seeds.append(rank-1)
        if len(drop_seeds) >= len(entry['data']):
            continue
        rank_basis = [min(_['objective']) for _ in entry['data']]
        ranks = np.argsort(rank_basis)
        new_entry_data = [entry['data'][_] for _ in ranks if _ not in drop_seeds]
        entry['data'] = new_entry_data
    return data

def combine_seeds(data, args):
    combined_data = []
    offset = 0
    for nentry, entry in enumerate(data):
        new_data = {'name': entry['name'], 'type': entry['type'], 'fname': entry['fname']}
        if entry['type'] == 'pca':
            # PCA requires special data combination beyond this point
            pca = pd.concat(entry['data'])
            offset += len(entry['data']) - 1
            other = [_ for _ in ['objective', 'predicted', 'elapsed_sec'] if _ in pca.columns]
            # Maintain proper column order despite using sets
            permitted = set(pca.columns).difference(set(other))
            params = [_ for _ in pca.columns if _ in permitted]
            # Find the duplicate indices to combine, then grab the parameter values of these unique duplicated values
            duplicate_values = pca.drop(columns=other)[pca.drop(columns=other).duplicated()].to_numpy()
            # BIG ONE HERE
            # NP.ALL() is looking for and'd columns matching a duplicate value for EACH column over the rows
            # Then we get the FULL rows from the original set of matches and GROUPBY params without resetting the index
            # This allows us to MEAN the remaining columns but have a DataFrame object come out, ie the reduced DataFrame
            # for all duplicates of this particular duplicated set of parameters
            frame_list = [pca[np.all([(pca[k]==v) for k,v in zip(params, values)], axis=0)].groupby(params, as_index=False).mean() for values in duplicate_values]
            # We then add the unique values (keep=False means ALL duplicates are excluded) to ensure data isn't deleted
            reconstructed = pd.concat([pca.drop_duplicates(subset=params, keep=False)]+frame_list).reset_index()
            # Trim points by top% and rerank
            if args.pca_tops is not None and args.pca_tops != []:
                # Get cutoff for this entry (NEAREST actual data)
                quants = reconstructed['objective'].quantile(args.pca_tops[nentry+offset], interpolation='nearest')
                # Make a new frame of top values at/above this cutoff
                reconstructed = reconstructed[reconstructed['objective'] >= quants].drop(columns='index').reset_index()
            # Trim points by quantile IF pca points is not None
            elif args.pca_points is not None and args.pca_points != []:
                # Use NEAREST (actual data) quantiles from range 0 to 1
                quants = reconstructed['objective'].quantile([_/(args.pca_points-1) for _ in range(args.pca_points)], interpolation='nearest')
                # Construct new frame consisting of only these quantile values
                # Drop the redundant 'index' column from it getting merged in there as well
                reconstructed = pd.concat([reconstructed[reconstructed['objective'] == q] for q in quants]).drop(columns='index').reset_index()
            new_data['data'] = reconstructed
            combined_data.append(new_data)
            continue
        elif entry['type'] == 'xfer':
            new_data['data'] = pd.concat([_[['source_objective','target_objective','source_size','target_size']] for _ in entry['data']])
            # NORMALIZE Y AXIS VALUES
            # GLOBAL NORM
            #tgt = new_data['data']['target_objective']
            #new_data['data']['target_objective'] = (tgt - min(tgt)) / (max(tgt)-min(tgt))
            # PER TARGET SIZE NORM
            new_data['data']['target_objective'] = new_data['data'].groupby('target_size').transform(lambda x: (x - x.min())/(x.max()-x.min()))['target_objective']
            combined_data.append(new_data)
            continue
        # Change objective column to be the average
        # Add min, max, and stddev columns for each point
        objective_priority = ['objective', 'exe_time']
        objective_col = 0
        try:
            while objective_priority[objective_col] not in entry['data'][0].columns:
                objective_col += 1
            objective_col = objective_priority[objective_col]
        except IndexError:
            print(entry['data'])
            raise ValueError(f"No known objective in {entry['name']} with columns {entry['data'][0].columns}")
        last_step = np.full(len(entry['data']), np.inf)
        if args.x_axis == 'evaluation':
            n_points = max([max(_.index)+1 for _ in entry['data']])
            steps = range(n_points)
        else:
            seconds = pd.concat([_['elapsed_sec'] for _ in entry['data']])
            if args.synchronous:
                steps = seconds.groupby(seconds.index).mean()
                lookup_steps = [dict((agg,personal) for agg, personal in \
                                    zip(steps, seconds.groupby(seconds.index).nth(idx))) \
                                        for idx in range(len(entry['data']))]
            else:
                steps = sorted(seconds.unique())
                # Set "last" objective value for things that start later to their first value
                for idx, frame in enumerate(entry['data']):
                    if frame['elapsed_sec'][0] != steps[0]:
                        last_step[idx] = frame[objective_col][0]
            n_points = len(steps)
        new_columns = {'min': np.zeros(n_points),
                       'max': np.zeros(n_points),
                       'std_low': np.zeros(n_points),
                       'std_high': np.zeros(n_points),
                       'obj': np.zeros(n_points),
                       'exe': np.zeros(n_points),
                       'current': np.zeros(n_points),
                      }
        prev_mean = None
        for idx, step in enumerate(steps):
            # Get the step data based on x-axis needs
            if args.x_axis == 'evaluation':
                step_data = []
                for idx2, df in enumerate(entry['data']):
                    if step in df.index:
                        last_step[idx2] = df.iloc[step][objective_col]
                    # Drop to infinity if shorter than the longest dataframe
                    else:
                        last_step[idx2] = np.inf
                    step_data.append(last_step[idx2])
            elif args.synchronous:
                step_data = []
                for idx2, df in enumerate(entry['data']):
                    try:
                        local_step = df[df['elapsed_sec'] == lookup_steps[idx2][step]].index[0]
                        last_step[idx2] = df.iloc[local_step][objective_col]
                    except (KeyError, IndexError):
                        pass
                    step_data.append(last_step[idx2])
            else:
                step_data = []
                for idx2, df in enumerate(entry['data']):
                    # Get objective value in the row where the step's elapsed time exists
                    lookup_index = df[objective_col][df[df['elapsed_sec'] == step].index]
                    if not lookup_index.empty:
                        last_step[idx2] = lookup_index.tolist()[0]
                    # Always add last known value (may have just been updated)
                    step_data.append(last_step[idx2])
            # Make data entries for new_columns, ignoring NaN/Inf values
            if len(step_data) == 1:
                new_columns['current'][idx] = step_data[0]
                new_columns['obj'][idx] = step_data[0]
                new_columns['exe'][idx] = step
                new_columns['min'][idx] = step_data[0]
                new_columns['max'][idx] = step_data[0]
                new_columns['std_low'][idx] = 0.0
                new_columns['std_high'][idx] = 0.0
                continue
            else:
                finite = [_ for _ in step_data if np.isfinite(_)]
                if len(finite) == 0:
                    new_columns['current'][idx] = np.nan
                    new_columns['obj'][idx] = np.nan
                    new_columns['exe'][idx] = step
                    new_columns['min'][idx] = np.nan
                    new_columns['max'][idx] = np.nan
                    new_columns['std_low'][idx] = np.nan
                    new_columns['std_high'][idx] = np.nan
                    continue
                mean = np.mean(finite)
                trimmed = entry['fname'] in args.trim
                if 'old_objective' in entry['data'][0].columns:
                    new_columns['current'][idx] = np.mean([_['old_objective'].iloc[idx] for _ in entry['data']])
                else:
                    new_columns['current'][idx] = mean
                if not trimmed or new_data['type'] != 'best' or prev_mean is None or mean != prev_mean:
                    new_columns['obj'][idx] = mean
                    prev_mean = mean
                    new_columns['exe'][idx] = step
                    if args.x_axis == 'evaluation' and args.log_x:
                        new_columns['exe'][idx] = step+1
                    new_columns['min'][idx] = min(finite)
                    new_columns['max'][idx] = max(finite)
                    if new_data['type'] == 'best':
                        new_columns['std_low'][idx] = new_columns['obj'][idx]-min(finite)
                        new_columns['std_high'][idx] = max(finite)-new_columns['obj'][idx]
                    else:
                        stddev = np.std(finite)
                        new_columns['std_low'][idx] = stddev
                        new_columns['std_high'][idx] = stddev
        # Make new dataframe
        new_data['data'] = pd.DataFrame(new_columns).sort_values('exe')
        #new_data['data'] = new_data['data'][new_data['data']['obj'] > 0]
        combined_data.append(new_data)
    # Perform PCA fitting
    fittable = []
    for entry in combined_data:
        if entry['type'] == 'pca':
            try:
                fittable.append(entry['data'].drop(columns='predicted'))
            except KeyError:
                fittable.append(entry['data'])
    if len(fittable) > 0:
        import importlib, skopt
        problem, attr = args.pca_problem.rsplit('.',1)
        module = importlib.import_module(problem)
        space = module.__getattr__(attr).input_space
        skopt_space = skopt.space.Space(space)
        # Transform data. Non-objective/runtime should become vectorized. Objective should be ranked
        regressions, rankings = [], []
        for data in fittable:
            parameters = data.loc[:, space.get_hyperparameter_names()]
            other = data.loc[:, [_ for _ in data.columns if _ not in space.get_hyperparameter_names()]]
            x_parameters = skopt_space.transform(parameters.astype('str').to_numpy())
            rankdict = dict((idx,rank+1) for (rank, idx) in zip(range(len(other['objective'])),
                    np.argsort(((-1)**args.max_objective) * np.asarray(other['objective']))))
            other.loc[:, ('objective')] = [rankdict[_] / len(other['objective']) for _ in other['objective'].index]
            regressions.append(x_parameters)
            rankings.append(other['objective'])
        if args.pca_algorithm == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
        else:
            from sklearn.manifold import TSNE
            pca = TSNE(n_components=2)
        pca_values = pca.fit_transform(np.vstack(regressions))
        # Re-assign over data
        pca_idx, combined_idx = 0, 0
        for regs, rerank in zip(regressions, rankings):
            required_idx = len(regs)
            new_frame = {'x': pca_values[pca_idx:pca_idx+required_idx,0],
                         'y': pca_values[pca_idx:pca_idx+required_idx,1],
                         'z': rerank}
            new_frame = pd.DataFrame(new_frame)
            pca_idx += required_idx
            combined_data[combined_idx]['data'] = new_frame
            combined_idx += 1
    # Find top val
    if args.top is None:
        top_val = None
    else:
        if args.global_top:
            top_val = np.quantile(pd.concat([_['data']['obj'] for _ in combined_data]), q=args.top)
        else:
            top_val = {_['name']: np.quantile(_['data']['obj'], q=args.top) for _ in combined_data}
    return combined_data, top_val

def load_all(args):
    legend_title = None
    data = []
    inv_names = []
    shortlist = []
    if args.inputs is not None:
        # Load all normal inputs
        for fname in args.inputs:
            #print(f"Load [Input]: {fname}")
            try:
                fd = pd.read_csv(fname)
            except IOError:
                print(f"WARNING: Could not open {fname}, removing from 'inputs' list")
                continue
            # Drop unnecessary parameters
            d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
            # Sometimes the objective is reported as exactly 1.0, which indicates inability to run that point.
            # Discard such rows when loading
            failure_rows = np.where(d['objective'].to_numpy()-1==0)[0]
            d.loc[failure_rows,'objective'] = np.nan
            if args.drop_overhead:
                first_non_nan_idx = list(~np.isnan(d['objective'])).index(True)
                d['elapsed_sec'] -= min(d['elapsed_sec'].iloc[0], (d['elapsed_sec'].iloc[1:].to_numpy() - d['elapsed_sec'].iloc[:-1].to_numpy())[first_non_nan_idx])
                #d['elapsed_sec'] -= d['elapsed_sec'].iloc[first_non_nan_idx]-d['objective'].iloc[first_non_nan_idx]
            if args.as_speedup_vs is not None:
                d['objective'] = args.as_speedup_vs / d['objective']
            # Potentially add assumption that at step/t=0 the objective is 1.0
            if args.tzero:
                d = pd.concat((pd.DataFrame({'objective': args.tobjective, 'elapsed_sec': 0.0},index=[0]),d)).reset_index(drop=True)
            name, directory, legend_title = make_seed_invariant_name(fname, args)
            fullname = directory+'.'+name
            if fullname in inv_names:
                idx = inv_names.index(fullname)
                # Just put them side-by-side for now
                data[idx]['data'].append(d)
            else:
                data.append({'name': fullname, 'data': [d], 'type': 'input',
                             'fname': fname, 'dir': directory})
                inv_names.append(fullname)
                shortlist.append(name)
    # Drop directory from names IF only represented once
    for (name, fullname) in zip(shortlist, inv_names):
        if shortlist.count(name) == 1:
            idx = inv_names.index(fullname)
            data[idx]['name'] = name
    idx_offset = len(data)
    inv_names = []
    shortlist = []
    # Load PCA inputs
    if args.pca is not None and args.pca != []:
        # Load all normal inputs
        for fname in args.pca:
            #print(f"Load [PCA]: {fname}")
            try:
                d = pd.read_csv(fname)
            except IOError:
                print(f"WARNING: Could not open {fname}, removing from 'pca' list")
                continue
            if args.drop_overhead:
                d['elapsed_sec'] -= d['elapsed_sec'].iloc[0]-d['objective'].iloc[0]
            if args.as_speedup_vs is not None:
                d['objective'] = args.as_speedup_vs / d['objective']
            name, directory, legend_title = make_seed_invariant_name(fname, args)
            fullname = directory+'.'+name
            if fullname in inv_names:
                idx = inv_names.index(fullname)
                # Just put them side-by-side for now
                data[idx]['data'].append(d)
            else:
                data.append({'name': fullname, 'data': [d], 'type': 'pca',
                             'fname': fname, 'dir': directory})
                inv_names.append(fullname)
                shortlist.append(name)
    # Drop directory from names IF only represented once
    for (name, fullname) in zip(shortlist, inv_names):
        if shortlist.count(name) == 1:
            idx = inv_names.index(fullname)
            data[idx]['name'] = name
    # Load xfer inputs
    idx_offset = len(data)
    inv_names = []
    shortlist = []
    if args.xfers is not None:
        # Load all force transfer inputs
        for fname in args.xfers:
            #print(f"Load [XFER]: {fname}")
            try:
                d = pd.read_csv(fname)
            except IOError:
                print(f"WARNING: Could not open {fname}, removing from 'xfer' list")
                continue
            name, directory, legend_title = make_seed_invariant_name(fname, args)
            fullname = directory+'.'+name
            if fullname in inv_names:
                idx = inv_names.index(fullname)
                # Just put them side-by-side for now
                data[idx]['data'].append(d)
            else:
                data.append({'name': fullname, 'data': [d], 'type': 'xfer',
                             'fname': fname, 'dir': directory})
                inv_names.append(fullname)
                shortlist.append(name)
    # Drop directory from names IF only represented once
    for (name, fullname) in zip(shortlist, inv_names):
        if shortlist.count(name) == 1:
            idx = inv_names.index(fullname)
            data[idx]['name'] = name
    # Load best-so-far inputs
    idx_offset = len(data) # Best-so-far have to be independent of normal inputs as the same file
                           # may be in both lists, but it should be treated by BOTH standards if so
    inv_names = []
    shortlist = []
    if args.bests is not None:
        for fname in args.bests:
            #print(f"Load [Best]: {fname}")
            try:
                fd = pd.read_csv(fname)
            except IOError:
                print(f"WARNING: Could not open {fname}, removing from 'bests' list")
                continue
            # Drop unnecessary parameters
            d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
            if args.drop_overhead:
                d['elapsed_sec'] -= d['elapsed_sec'].iloc[0]-d['objective'].iloc[0]
            if args.as_speedup_vs is not None:
                d['objective'] = args.as_speedup_vs / d['objective']
            # Transform into best-so-far dataset
            for col in ['objective', 'exe_time']:
                if col in d.columns:
                    old_vals = d[col].tolist()
                    d['old_'+col] = old_vals
                    if args.max_objective:
                        d[col] = [max(d[col][:_+1]) for _ in range(0,len(d[col]))]
                    else:
                        d[col] = [min(d[col][:_+1]) for _ in range(0,len(d[col]))]
            name, directory, legend_title = make_seed_invariant_name(fname, args)
            if not args.clean_names:
                name = "best_"+name
                fullname = directory+'.'+name
            else:
                fullname = name
            if fullname in inv_names:
                idx = inv_names.index(fullname)
                # Just put them side-by-side for now
                data[idx_offset+idx]['data'].append(d)
                if type(data[idx_offset+idx]['fname']) is str:
                    data[idx_offset+idx]['fname'] = [data[idx_offset+idx]['fname'], fname]
                else:
                    data[idx_offset+idx]['fname'].append(fname)
            else:
                data.append({'name': fullname, 'data': [d], 'type': 'best',
                             'fname': fname, 'dir': directory})
                inv_names.append(fullname)
                shortlist.append(name)
    # Drop directory from names IF only represented once
    for (name, fullname) in zip(shortlist, inv_names):
        if shortlist.count(name) == 1:
            idx = inv_names.index(fullname)
            data[idx]['name'] = name
    idx_offset = len(data) # Best-so-far have to be independent of normal inputs as the same file
                           # may be in both lists, but it should be treated by BOTH standards if so
    inv_names = []
    if args.baseline_best is not None:
        for fname in args.baseline_best:
            #print(f"Load [Baseline]: {fname}")
            try:
                fd = pd.read_csv(fname)
            except IOError:
                print(f"WARNING: Could not open {fname}, removing from 'baseline_best' list")
                continue
            # Find ultimate best value to plot as horizontal line
            d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
            if args.drop_overhead:
                d['elapsed_sec'] -= d['elapsed_sec'].iloc[0]-d['objective'].iloc[0]
            if args.as_speedup_vs is not None:
                d['objective'] = args.as_speedup_vs / d['objective']
            # Transform into best-so-far dataset
            objval = None
            for col in ['objective', 'exe_time']:
                if col in d.columns:
                    if args.max_objective:
                        d[col] = [max(d[col][:_+1]) for _ in range(0, len(d[col]))]
                        objval = max(d[col])
                    else:
                        d[col] = [min(d[col][:_+1]) for _ in range(0, len(d[col]))]
                        objval = min(d[col])
                    if not args.clean_names:
                        name = "baseline_"+make_baseline_name(fname, args, d, col)[0]
                    matchname = 'baseline_'+make_seed_invariant_name(fname, args)[0]
                    break
            if matchname in inv_names:
                idx = inv_names.index(matchname)
                # Replace if improvement
                if args.max_objective:
                    if objval > data[idx_offset+idx]['objval']:
                        data[idx_offset+idx]['data'][0] = d
                else:
                    if objval < data[idx_offset+idx]['objval']:
                        data[idx_offset+idx]['data'][0] = d
            else:
                data.append({'name': name, 'type': 'baseline',
                             'matchname': matchname,
                             'objval': objval,
                             'data': [d],
                             'fname': fname})
                inv_names.append(matchname)
    # Fix across seeds
    return *combine_seeds(drop_seeds(data, args), args), legend_title

def prepare_fig(args):
    fig, ax = plt.subplots(figsize=tuple(args.fig_dims))
    fig.set_tight_layout(True)
    if args.top is None:
        if args.pca is not None and args.pca != []:
            name = args.pca_algorithm
        else:
            name = "plot"
    else:
        name = "competitive"
    return fig, ax, name

def alter_color(color_tup, ratio=0.5, brighten=True):
    return tuple(np.clip([ratio*(_+((-1)**(1+int(brighten)))) for _ in color_tup],0,1))

def plot_source(fig, ax, breaks, idx, source, args, ntypes, top_val=None):
    makeNew = False
    rfig = None
    rax = None
    data = source['data']
    if type(ax) is not list:
        ax_list = [ax]
    else:
        ax_list = ax
    if breaks is not None:
        fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
        rfig = fig
        fig.subplots_adjust(hspace=0.1)
        ax_list = [ax1,ax2]
        rax = ax_list
    # Color help
    colors = [mcolors.to_rgb(_['color']) for _ in list(plt.rcParams['axes.prop_cycle'])]
    color = colors[idx % len(colors)]
    color_maps = ['Oranges', 'Blues', 'Greens', 'Purples', 'Reds']
    #color_maps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    #color_maps = [_ for _ in plt.cm._cmap_registry.keys() if not _.endswith('_r')]
    color_map = color_maps[idx % len(color_maps)]
    #color_map = 'Reds'
    if source['type'] == 'pca':
        plt.scatter(data['x'], data['y'], c=data['z'], cmap=color_map, label=source['name'])#, labelcolor=color_map.lower().rstrip('s'))
        print(f"PCA Scatter {source['name']}")
    elif source['type'] == 'xfer':
        for target_line, color in zip(set(data['target_size']), colors):
            subset_data = data[data['target_size'] == target_line]
            plt.plot(subset_data['source_size'], subset_data['target_objective'], c=color, label=str(target_line), marker='.',markersize=12)
            print(f"XFER Plot {target_line}")
        makeNew = True
    elif top_val is None:
        for axes in ax_list:
            # Shaded area = stddev
            # Prevent <0 unless arg says otherwise
            if args.stddev:
                lower_bound = pd.DataFrame(data['obj']-data['std_low'])
                if not args.below_zero:
                    lower_bound = lower_bound.applymap(lambda x: max(x,0))
                lower_bound = lower_bound[0]
                axes.fill_between(data['exe'], lower_bound, data['obj']+data['std_high'],
                                label=f"Stddev {source['name']}",
                                alpha=0.4,
                                color=alter_color(color), zorder=-1)
                print(f"STDDEV Fill-Between {source['name']}")
            # Shaded area = current progress
            if args.current:
                axes.fill_between(data['exe'], data['current'], data['obj'],
                                label=f"Current {source['name']}",
                                alpha=0.4,
                                color=alter_color(color), zorder=-1)
                print(f"CURRENT Fill-Between {source['name']}")
            # Main line = mean
            mpl_marker = 'o'
            if len(data['obj']) > 1:
                cutoff = data['obj'].to_list().index(max(data['obj'][~np.isnan(data['obj'])]))
                axes.plot([0]+data['exe'][:min(cutoff+1, len(data))].tolist(), [1]+data['obj'][:min(cutoff+1,len(data))].tolist(),
                        label=f"Mean {source['name']}" if ntypes > 1 else source['name'],
                        marker=mpl_marker, color=color, zorder=1)
                print(f"MEAN Plot {source['name']}")
                if not args.cutoff:
                    axes.plot(data['exe'][cutoff:].tolist(), data['obj'][cutoff:].tolist(),
                            marker=mpl_marker, color=color, zorder=1)
                    print("\tMEAN-CUTOFF Plot")
            else:
                x_lims = [int(v) for v in axes.get_xlim()]
                x_lims[0] = max(0, x_lims[0])
                if x_lims[1]-x_lims[0] == 0:
                    x_lims[1] = x_lims[0]+1
                axes.plot(x_lims, [data['obj'], data['obj']],
                        label=f"Mean {source['name']}" if ntypes > 1 else source['name'],
                        marker=mpl_marker, color=color, zorder=1)
                print(f"MEAN Plot {source['name']}")
            # Flank lines = min/max
            if args.minmax:
                axes.plot(data['exe'], data['min'], linestyle='--',
                        label=f"Min/Max {source['name']}",
                        color=alter_color(color, brighten=False), zorder=0)
                print(f"MIN Plot {source['name']}")
                axes.plot(data['exe'], data['max'], linestyle='--',
                        color=alter_color(color, brighten=False), zorder=0)
                print(f"MAX Plot {source['name']}")
    else:
        # Make new Y that increases by 1 each time you beat the top val (based on min or max objective)
        if args.global_top:
            top = top_val
        else:
            top = top_val[source['name']]
        new_y = []
        counter = 0
        for val in data['obj']:
            if args.max_objective and val > top:
                counter += 1
            if not args.max_objective and val < top:
                counter += 1
            new_y.append(counter)
        ax.plot(data['exe'], new_y, label=source['name'],
                marker='.', color=color, zorder=1)
        print(f"TOP_VAL Plot {source['name']}")
    if args.budget is not None:
        if args.max_objective:
            y_height = max(source['data'].iloc[:args.budget].obj)
        else:
            y_height = min(source['data'].iloc[:args.budget].obj)
        x_width = data['exe'][args.budget-1]
        # Have to collect current x/y bounds or the plot gets rescaled!
        for ax in ax_list:
            budget_line = ax.plot([0,x_width],[y_height, y_height],color=color, linestyle='dotted')
            budget_vline = ax.vlines(x_width, 0,y_height, color=color, linestyle='dotted')
    return makeNew, rfig, rax

def text_analysis(all_data, args):
    best_results = {}
    for source in all_data:
        data = source['data']
        # Announce the line's best result
        if args.max_objective:
            best_y = max(data['obj'])
        else:
            best_y = min(data['obj'])
        best_x = data['exe'].iloc[data['obj'].to_list().index(best_y)]
        best_results[source['name']] = {'best_y': best_y,
                                        'best_x': best_x}
    for k,v in best_results.items():
        print(f"{k} BEST RESULT: {v['best_y']} at x = {v['best_x']}")
        if 'DEFAULT' not in k:
            best_results[k]['advantage'] = 0
        for k2,v2 in best_results.items():
            if k2 == k:
                continue
            if args.max_objective:
                improvement = v['best_y'] / v2['best_y']
            else:
                improvement = v2['best_y'] / v['best_y']
            improved = improvement > 1
            if not improved:
                improvement = 1 / improvement
            print("\t"+f"{'Better than' if improved else 'Worse than'} {k2}'s best by {improvement}")
            # Speedup ALWAYS goes this way
            speedup = v2['best_x'] / v['best_x']
            speed = speedup > 1
            if not speed:
                speedup = 1 / speedup
            print("\t\t"+f"{'Speedup' if speed else 'Slowdown'} to best solution of {speedup}")
            if not improved:
                improvement *= -1
            if not speed:
                speedup *= -1
            print("\t\t"+f"Advantage: {improvement + speedup}")
            if 'DEFAULT' not in k:
                best_results[k]['advantage'] += improvement + speedup
    winners, advantages = [], []
    for k,v in best_results.items():
        if 'advantage' not in v.keys():
            continue
        winners.append(k)
        advantages.append(v['advantage'])
        print(f"{k} sum advantage {v['advantage']}")
    advantage = max(advantages)
    winner = winners[advantages.index(advantage)]
    print(f"Most advantaged {winner} with sum advantage {advantage}")

def main(args):
    data, top_val, legend_title = load_all(args)
    fig, ax, name = prepare_fig(args)
    figures = [fig]
    axes = [ax]
    breaks = []
    names = [name]
    ntypes = len(set([_['type'] for _ in data]))
    if not args.no_text:
        text_analysis(data, args)
    if not args.no_plots:
        # Determine if a vertical break is needed and where
        y_data = np.vstack([d['data'].obj.to_numpy() for d in data])
        # Never need vertical breaks if max speedup is less than 3
        if np.max(y_data) > 40:
            y_flat = y_data.ravel()
            y_sort = np.argsort(-y_flat) # Speedups are positive, so mult -1 to get descending order
            z_score = (y_flat[y_sort]-y_flat.mean())/np.std(y_flat)
            outliers = np.where(z_score>1)[0]
            # Break needed
            if len(outliers) > 0:
                floor_break = min(np.floor(y_flat[y_sort[outliers]]).astype(int))
                ceil_break = max(np.ceil(y_flat[y_sort[outliers]+1]).astype(int))
                breaks.append((ceil_break, floor_break))
                for idx in range(y_data.shape[1]-1):
                    breaks.append(None)
            else:
                breaks.extend([None for _ in range(y_data.shape[1])])
        else:
            breaks.extend([None for _ in range(y_data.shape[1])])
        for idx, source in enumerate(data):
            print(f"plot {source['name']} based upon {source['fname']}")
            newfig, fig, ax = plot_source(figures[-1], axes[-1], breaks[idx], idx, source, args, ntypes, top_val)
            # Check if figure replaced by the call
            if fig is not None and ax is not None:
                figures[-1] = fig
                axes[-1] = ax
            # XFER may generate additional figures
            if newfig:
                if names[-1] == 'plot':
                    names[-1] = source['name']
                fig, ax, name = prepare_fig(args)
                figures.append(fig)
                axes.append(ax)
                names.append(name)
        if newfig:
            del figures[-1], axes[-1], names[-1]
        min_mul = 0.9983
        max_mul = 1+(1-min_mul)
        finite_x = [d['data'].exe[~np.isnan(d['data'].exe)] for d in data]
        finite_y = [d['data'].obj[~np.isnan(d['data'].obj)] for d in data]
        xlims = [min_mul * min([min(x) for x in finite_x]),
                 max_mul * max([max(x) for x in finite_x])]
        ylims = [min_mul * min([min(y) for y in finite_y]),
                 max_mul * max([max(y) for y in finite_y])]
        for (fig, ax, name) in zip(figures, axes, names):
            # make x-axis data
            if args.pca is not None and args.pca != []:
                xname = f'{args.pca_algorithm.upper()} dimension 1'
            else:
                if args.x_axis == "evaluation":
                    xname = "Evaluation #"
                elif args.x_axis == "walltime":
                    xname = "Elapsed Time (seconds)"
            # make y-axis data
            if args.pca is not None and args.pca != []:
                yname = f'{args.pca_algorithm.upper()} dimension 2'
            else:
                if top_val is None:
                    if args.as_speedup_vs is not None:
                        yname = "Speedup over Baseline"
                        #yname = "Speedup (over -O3 -polly)"
                    else:
                        yname = "Objective"
                else:
                    if args.global_top:
                        yname = f"# Configs with top {round(100*args.top,1)}% result = {round(top_val,4)}"
                    else:
                        yname = f"# Configs with top {round(100*args.top,1)}% result per technique"
            if not isinstance(ax, matplotlib.axes.Axes):
                # Handle limiting and splitting here
                ax[1].set_ylim(max(0,min_mul * np.min(y_data)),breaks[0][0])
                above_break = y_flat[np.where(y_flat > breaks[0][1])[0]]
                other_span = breaks[0][0] - max(0, min_mul * np.min(y_data))
                if len(above_break) == 1:
                    ax[0].set_ylim(above_break - (other_span / 2), above_break + (other_span / 2))
                elif np.max(above_break)-np.min(above_break) < other_span:
                    ax[0].set_ylim(np.min(above_break) - (other_span / 2), np.max(above_break) + (other_span / 2))
                else:
                    ax[0].set_ylim(min(np.min(above_break), max_mul * breaks[0][1]), max(np.max(above_break), min_mul * np.ceil(np.max(y_data))))
                # SAFETY CHECK FOR POINTS SWALLOWED BY THE BREAK
                invisible = y_flat[np.logical_and(np.where(y_flat < ax[0].get_ylim()[0], True, False), np.where(y_flat > ax[1].get_ylim()[1], True, False))]
                if len(invisible) > 0:
                    print(f" !! WARNING: Vertical break hides {len(invisible)} points from display !! Find and fix the issue !!")
                # Hide spines
                # ax[1] == ax2
                # ax[0] == ax1
                ax[0].spines.bottom.set_visible(False)
                ax[1].spines.top.set_visible(False)
                ax[0].xaxis.tick_top()
                ax[0].tick_params(labeltop=False)
                d = 0.5
                kwargs = dict(marker=[(-1,-d),(1,d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax[0].plot([0,1],[0,0], transform=ax[0].transAxes, **kwargs)
                ax[1].plot([0,1],[1,1], transform=ax[1].transAxes, **kwargs)
                # Normal axes treatment
                ax[1].set_xlabel(xname)
                # Have to shift the y-axis label
                if 'xl' in args.output:
                    ax[0].set_box_aspect(1/8)
                    ylabel = ax[1].set_ylabel(yname, labelpad=10)
                    ylabel.set_horizontalalignment('left')
                    ylabel.set_position((0,0.15))
                else:
                    ax[0].set_box_aspect(1/5)
                    ylabel = ax[1].set_ylabel(yname, labelpad=10)
                    ylabel.set_horizontalalignment('left')
                    ylabel.set_position((0,0.15))
                if args.log_x:
                    ax[0].set_xscale("symlog")
                if args.log_y:
                    ax[0].set_yscale("symlog")
                    ax[1].set_yscale("symlog")
                ax[1].set_xlim(xlims)
                ax[0].grid()
                ax[1].grid()
                if args.legend is not None:
                    """
                    if len(ax.collections) > 0:
                        colors = [_.cmap.name.lower().rstrip('s') for _ in ax.collections]
                        leg_handles = [matplotlib.lines.Line2D([0],[0],
                                                marker='o',
                                                color='w',
                                                label=l.get_label(),
                                                markerfacecolor=c,
                                                markersize=8,
                                                ) for c,l in zip(colors,ax.collections)]
                        ax.legend(handles=leg_handles, loc=" ".join(args.legend), title=legend_title)
                    """
                    # This is a hack but I don't know a better way to do it for now
                    loc = "upper right"
                    kwargs = {'borderaxespad': 0}
                    ax_idx = 0
                    if 'xl' in args.output:
                        loc = "lower left"
                        kwargs.update({'bbox_to_anchor': (0,0.48)})
                        ax_idx = 1
                    ax[ax_idx].legend(loc=loc, title=legend_title, **kwargs)
            else:
                ax.set_xlabel(xname)
                ax.set_ylabel(yname)
                if args.log_x:
                    ax.set_xscale("symlog")
                if args.log_y:
                    ax.set_yscale("symlog")
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                ax.grid()
                if args.legend is not None:
                    """
                    if len(ax.collections) > 0:
                        colors = [_.cmap.name.lower().rstrip('s') for _ in ax.collections]
                        leg_handles = [matplotlib.lines.Line2D([0],[0],
                                                marker='o',
                                                color='w',
                                                label=l.get_label(),
                                                markerfacecolor=c,
                                                markersize=8,
                                                ) for c,l in zip(colors,ax.collections)]
                        ax.legend(handles=leg_handles, loc=" ".join(args.legend), title=legend_title)
                    """
                    ax.legend(loc=" ".join(args.legend), title=legend_title)
            if not args.show:
                fig.tight_layout()
                fig.savefig("_".join([args.output,name])+f'.{args.format}', format=args.format, bbox_inches='tight')
    if args.show:
        plt.show()

if __name__ == '__main__':
    main(parse(build()))

