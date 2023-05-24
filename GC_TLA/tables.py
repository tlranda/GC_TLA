import argparse, os, itertools
import pandas as pd, numpy as np
from contextlib import nullcontext

def build():
    prs = argparse.ArgumentParser()
    subparsers = prs.add_subparsers(dest='subparser_name')
    # RUN THESE FIRST
    task_parser = subparsers.add_parser('task', help='Determine table data for a given benchmark task')
    task_inputs = task_parser.add_argument_group('Inputs')
    task_inputs.add_argument('--inputs', type=str, nargs='+', required=True, help="Data files to read")
    task_inputs.add_argument('--ignore', type=str, nargs='*', help="Data files to ignore if globbed by inputs")
    task_inputs.add_argument('--as-speedup-vs', type=str, default=None, help="Convert metrics to speedup based on this float or value derived from a CSV file with this name")
    task_inputs.add_argument('--budget', type=int, default=None, help="Report budget at this number of evaluations")
    task_options = task_parser.add_argument_group('Options')
    task_options.add_argument("--drop-overhead", action="store_true", help="Attempt to remove initialization overhead time in seconds if this argument is specified")
    task_options.add_argument("--max-objective", action="store_true", help="Objective is MAXIMIZE not MINIMIZE (default MINIMIZE)")
    task_options.add_argument("--round", type=int, default=None, help="Round values to this many decimal places (default no rounding)")
    task_options.add_argument("--quiet", action="store_true", help="Silence normal output (does not prevent writing to output files) when specified")
    task_options.add_argument("--show-seed", action='store_true', help="Indicate seed of file source if this argument is specified")
    task_options.add_argument("--average-seeds", action='store_true', help="Average performance across seeds if this argument is specified")
    task_outputs = task_parser.add_argument_group('Outputs')
    task_outputs.add_argument("--output-name", default=None, help="Supply a path to save results to (none saved if not specified)")
    task_outputs.add_argument("--overwrite", action='store_true', help="Override clobber protection for output file path")

    # COMBINE DISTINCT RESULTS FROM SEPARATE 'task' RUNS
    collate_parser = subparsers.add_parser('collate', help='Present table data for a collection of benchmark tasks')
    collate_parser.add_argument('--inputs', type=str, nargs='+', required=True, help="Task outputs from earlier runs of <task> command of this script")
    collate_parser.add_argument("--round", type=int, default=None, help="Round values to this many decimal places (default no rounding)")
    collate_parser.add_argument("--max-objective", action="store_true", help="Objective is MAXIMIZE not MINIMIZE (default MINIMIZE)")
    collate_parser.add_argument("--latex", action="store_true", help="Output as latex instead of Pandas Dataframe string when specified")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.subparser_name == 'task':
        if args.ignore is not None:
            allowed = []
            for fname in args.inputs:
                if fname not in args.ignore:
                    allowed.append(fname)
            args.inputs = allowed
        # Special ordering helps with table output like paper
        ordered = []
        for selection in ['GaussianCopula', 'rf', 'gptune']:
            for fname in args.inputs:
                if selection in fname and fname not in ordered:
                    ordered.append(fname)
        # Pick up everything that wasn't caught in special ordering
        for fname in args.inputs:
            if fname not in ordered:
                ordered.append(fname)
        args.inputs = ordered
        if args.as_speedup_vs is not None:
            try:
                args.as_speedup_vs = float(args.as_speedup_vs)
            except ValueError:
                args.as_speedup_vs = pd.read_csv(args.as_speedup_vs).iloc[0]['objective']
    return args

def make_seed_invariant_name(name, args):
    directory = os.path.dirname(name)
    name = os.path.basename(name)
    name_dot, ext = name.rsplit('.',1)
    if name_dot.endswith("_ALL"):
        name_dot = name_dot[:-4]
    try:
        base, seed = name_dot.rsplit('_',1)
        intval = int(seed)
        name = base
    except ValueError:
        if '.' in name:
            name, _ = name.rsplit('.',1)
        name = name.lstrip("_")
    else:
        if '.' in name:
            name, _ = name.rsplit('.',1)
    name = name.lstrip("_")
    return name, directory

def combine_seeds(data, args):
    combined_data = []
    offset = 0
    for nentry, entry in enumerate(data):
        new_data = {'name': entry['name'], 'type': entry['type'], 'fname': entry['fname']}
        # Change objective column to be the average
        objective_priority = ['objective', 'exe_time']
        objective_col = 0
        try:
            while objective_priority[objective_col] not in entry['data'][0].columns:
                objective_col += 1
            objective_col = objective_priority[objective_col]
        except IndexError:
            print(entry['data'])
            raise ValueError(f"No known objective in {entry['name']} with columns {entry['data'][0].columns}")
        n_points = max(map(len, entry['data']))
        n_seeds = len(entry['data'])
        objs = np.stack([entry['data'][_][objective_col].to_numpy() for _ in range(n_seeds)])
        # PERHAPS AVERAGED?
        objs = np.mean(objs, axis=0)
        if args.max_objective:
            seed_attribution = np.argmax(objs,axis=0)
        else:
            seed_attribution = np.argmin(objs,axis=0)
        #objs = objs[seed_attribution, np.arange(n_points)]
        new_columns = {'obj': objs,
                       'src_file': ['average' for _ in np.arange(n_points)],}
                       #'src_file': np.asarray(entry['fname'])[seed_attribution],}
        # Make new dataframe
        new_data['data'] = pd.DataFrame(new_columns)
        combined_data.append(new_data)
    return combined_data

def load_task_inputs(args):
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
                if not args.quiet:
                    print(f"WARNING: Could not open {fname}, removing from 'inputs' list")
                continue
            # Sometimes the objective is reported as exactly 1.0, which indicates inability to run that point.
            # Discard such rows when loading
            failure_rows = np.where(fd['objective'].to_numpy()-1==0)[0]
            fd.loc[failure_rows,'objective'] = np.nan
            # Drop unnecessary parameters
            d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
            if args.drop_overhead:
                d['elapsed_sec'] -= d['elapsed_sec'].iloc[0]-d['objective'].iloc[0]
            if args.as_speedup_vs is not None:
                d['objective'] = args.as_speedup_vs / d['objective']
            name, directory = make_seed_invariant_name(fname, args)
            fullname = directory+'.'+name
            if fullname in inv_names:
                idx = inv_names.index(fullname)
                # Just put them side-by-side for now
                data[idx]['data'].append(d)
                data[idx]['fname'].append(fname)
            else:
                data.append({'name': fullname, 'data': [d], 'type': 'input',
                             'fname': [fname], 'dir': directory})
                inv_names.append(fullname)
                shortlist.append(name)
    # Drop directory from names IF only represented once
    for (name, fullname) in zip(shortlist, inv_names):
        if shortlist.count(name) == 1:
            idx = inv_names.index(fullname)
            data[idx]['name'] = name
    # Fix across seeds
    return combine_seeds(data, args)

def table_analyze(data, args):
    # COLUMNS: GC First, GC Budget, GC Best, BO Best, GPTune Best
    with (open(args.output_name, 'w') if args.output_name is not None else nullcontext()) as FILE:
        if args.output_name is not None:
            FILE.write('GC_First,GC_Budget,GC_Budget_At,GC_Best,GC_Best_At,BO_Best,BO_Best_At,GPTune_Best,GPTune_Best_At\n')
        for eidx, entry in enumerate(data):
            if 'GaussianCopula' in entry['name']:
                # First result
                first_result = entry['data'].iloc[0]['obj']
                # Budget result
                if args.budget is None:
                    budget_result = None
                else:
                    if args.max_objective:
                        budget_idx = np.argmax(entry['data'].iloc[:args.budget]['obj'])
                    else:
                        budget_idx = np.argmin(entry['data'].iloc[:args.budget]['obj'])
                    budget_result = entry['data'].iloc[budget_idx]['obj']
            else:
                first_result = None
                budget_result = None
            if args.max_objective:
                best_idx = np.argmax(entry['data']['obj'])
            else:
                best_idx = np.argmin(entry['data']['obj'])
            best_result = entry['data'].iloc[best_idx]['obj']
            # IDXs + 1 to be the nth evaluation (1-indexed, rather than 0-indexed row identiifer)
            if args.output_name is not None:
                if first_result is not None:
                    FILE.write(f"{first_result},")
                if budget_result is not None:
                    FILE.write(f"{budget_result},{budget_idx+1},")
                FILE.write(f"{best_result},{best_idx+1}")
                if eidx != len(data)-1:
                    FILE.write(",")
            if not args.quiet:
                # Rounding on display only
                if args.round is not None:
                    if first_result is not None:
                        first_result = round(first_result, args.round)
                    if budget_result is not None:
                        budget_result = round(budget_result, args.round)
                    best_result = round(best_result, args.round)
                print(f"> {entry['name']} | {first_result if first_result is not None else ''} | "+\
                      f"{budget_result if budget_result is not None else ''} {'('+str(budget_idx+1)+')' if budget_result is not None else ''} | "+\
                      f"{best_result} ({best_idx+1})")
                print("\t\t"+f"Results based on: {entry['fname']}")
                print("\t\t"+f"Best result from: {entry['data'].iloc[best_idx]['src_file']}")
                if first_result is not None:
                    print("\t\t"+f"First result from: {entry['data'].iloc[0]['src_file']}")
                if budget_result is not None:
                    print("\t\t"+f"Budget result from: {entry['data'].iloc[budget_idx]['src_file']}")
        if args.output_name is not None:
            FILE.write('\n')

def get_best_of_row(row, col_subset, maximize_objective):
    # Return the column name in the subset that max/minimizes the objective best in this row

    # Due to argmin/argmax tiebreakers going to the lower index,
    # we take special care to reorder the row data so that the comparisons
    # make the most sense.

    # For instance, if GC_Budget > BO_Best && GC_Budget > GPTune_Best,
    # and our goal is to maximize the objective, we'd rather report
    # GC_Budget as the best even if GC_Best > GC_Budget.

    # The ordering and selection logic outlined here will always pick the
    # FIRST specified data for a technique (column prefix until reaching
    # an underscore character) that outperforms ALL other data for ALL
    # other techniques

    # In the unlikely event that the best objective values are identical,
    # the tiebreaker order DISADVANTAGES our technique by placing
    # BO and GPTune prior in the order, therefore minimizing their indices

    # Find the BEST result per row using argmin/argmax
    best_selector = np.argmax if maximize_objective else np.argmin
    # Split records based on their technique prefix
    starter_subs = sorted(set([_.split("_",1)[0] for _ in col_subset]))
    # Know we know how to sort each technique based on its prefix and get all options
    # Stack them in the order they appear left->right in CSV, so leftmost option is MOST
    # preferable
    grouped_columns = dict((k,[v for v in col_subset if v.startswith(k)]) for k in starter_subs)
    # Determine most number of options to check against
    depth = max(map(len,grouped_columns.values()))
    # Stack all options (if any list is SHORTER, repeat its LAST (best shot) option to pad length)
    valueStack = np.zeros((depth,len(starter_subs)))
    nameStack = np.empty((depth,len(starter_subs)),dtype=object)
    for idx in range(depth):
        nameStack[idx] = [_[min(idx, max(0,len(_)-1))] for _ in grouped_columns.values()]
        valueStack[idx] = row[nameStack[idx]]
    best_per_option = best_selector(valueStack,axis=1)
    # Pick the best, with MOST AGGRESSIVE tiebreaker when this technique
    # remains best at higher priority, even if that's not global best
    best_option = best_selector(valueStack[np.arange(depth),best_per_option])
    # There are cool numpy ways to do this, but given that the list isn't long and it takes
    # several API calls, plain Python is probably faster
    while best_option > 0 and best_per_option[best_option-1] == best_per_option[best_option]:
        best_option -= 1
    # Re-select what that option was
    best_idx = best_selector(valueStack[best_option])
    # Finalize for return
    return nameStack[best_option, best_idx]

def latex_prelude():
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{\\shepherdAlter{Polybench autotuning results after a maximum of 30 evaluations; "+
          "results are averaged across three repeated tuning attempts with unique seeds}{Autotuning "+
          "results after a maximum of 30 evaluations; results are averaged across three repeated "+
          "tuning attempts with unique seeds.}}")
    #print("\\caption{}")
    print("{\\scriptsize")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\multirow{3}{*}{} & \\multirow{3}{*}{Application} & \\multirow{3}{*}{Scale} & \\multicolumn{5}{c|}{Peak Speedup (\\# Evaluation Discovered)} \\\\\\cline{4-8}")
    print(" & & & \multicolumn{3}{c|}{GC} & \\multirow{2}{0.05\\linewidth}{BO Best} & \\multirow{2}{0.08\\linewidth}{GPTune Best} \\\\")
    print(" & & & 1$^{st}$ & Budget & Best & & \\\\")

def reorder_data(data, polybench_tasks, ecp_tasks):
    data = data.reset_index(drop=True)
    tasks = {}
    for idx, indication in enumerate(data.task):
        ind = indication.split(' ',1)[0]
        if ind not in tasks.keys():
            tasks[ind] = [idx]
        else:
            tasks[ind].append(idx)
    # Force sorting
    sort_order = []
    polybench_start = None
    n_polybench = 0
    ecp_start = None
    n_ecp = 0
    for task in polybench_tasks:
        if task in tasks.keys():
            sort_order.extend(tasks[task])
            n_polybench += len(tasks[task])
            if polybench_start is None:
                polybench_start = tasks[task][0]
    for task in ecp_tasks:
        if task in tasks.keys():
            sort_order.extend(tasks[task])
            n_ecp += len(tasks[task])
            if ecp_start is None:
                ecp_start = tasks[task][0]
    data = data.reindex(sort_order)
    return data, polybench_start, n_polybench, ecp_start, n_ecp, tasks

def latex_postlude():
    print("\\hline")
    print("\\end{tabular}")
    print("}")
    print("\\label{tbl:general_results}")
    print("\\end{table}")

def form_metric(row, metric, at_columns, arr):
    if row['best'] == metric:
        arr.append("\\cellcolor{blue!25} "+f"{row[metric]:.2f}")
    else:
        arr.append(f"{row[metric]:.2f}")
    if metric+"_At" in at_columns:
        arr[-1] += f" ({row[metric+'_At']})"
    return arr

def latexify(data,args):
    not_at_columns = [_ for idx,_ in enumerate(data.columns) if idx > 1 and not _.endswith('_At')]
    at_columns = [_ for _ in data.columns if _.endswith('_At')]
    latex_prelude()
    polybench_tasks = ['3mm','covariance','floyd_warshall','heat3d','lu','syr2k']
    ecp_tasks = ['amg','rsbench','sw4lite','xsbench']
    stylize = {'3mm': '3mm', 'covariance': "Cov.", 'floyd_warshall': "Floyd-W.", 'heat3d': "Heat3d", 'lu': "LU",
               'syr2k': "Syr2k", 'amg': "AMG", 'rsbench': "RSBench", 'xsbench': "XSBench", 'sw4lite': "SW4Lite"}
    data, polybench_start, n_polybench, ecp_start, n_ecp, tasks = reorder_data(data, polybench_tasks, ecp_tasks)
    seen_tasks = []
    for (idx, row) in data.iterrows():
        # Benchmark set indicator
        if polybench_start is not None and idx == polybench_start:
            print("\\hline")
            print("% Polybench")
            print("\\multirow{"+str(n_polybench)+"}{*}{\\rotatebox[origin=c]{90}{Polybench Kernels}} & ", end='')
        elif ecp_start is not None and idx == ecp_start:
            print("\\hline")
            print("% ECP")
            print("\\multirow{"+str(n_ecp)+"}{*}{\\rotatebox[origin=c]{90}{Exascale Computing Proxies}} & ", end='')
        else:
            print(" & ", end='')
        # Application indicator
        task_indicator = row['task'].split(' ',1)[0]
        if task_indicator not in seen_tasks:
            print("\\multirow{"+str(len(tasks[task_indicator]))+"}{*}{"+stylize[task_indicator]+"} & ", end='')
            seen_tasks.append(task_indicator)
        else:
            print(" & ", end='')
        # Scale indicator
        scale_indicator = row['task'].rsplit(' ',1)[1]
        print(scale_indicator+" & ", end='')
        # Collate the data and present it
        metrics = []
        gc_collate = []
        gc_count = 0
        for metric in not_at_columns:
            # Add this to GC collated column
            if metric.startswith("GC_"):
                gc_collate = form_metric(row, metric, at_columns, gc_collate)
                if gc_count == 2:
                    metrics.append(" & ".join(gc_collate))
                else:
                    gc_count += 1
            else:
                metrics = form_metric(row, metric, at_columns, metrics)
        print(" & ".join(metrics), end='')
        # Just row line between applications
        if scale_indicator == "XL":
            print(" \\\\\\cline{2-8}")
        else:
            print(" \\\\")
    latex_postlude()

def load_collate_inputs(args):
    data = None
    for fname in args.inputs:
        load = pd.read_csv(fname)
        not_at_columns = [_ for _ in load.columns if not _.endswith('_At')]
        # Determine best
        load.insert(0, 'best', [get_best_of_row(load.iloc[_], not_at_columns, args.max_objective) for _ in range(len(load))])
        # Rounding
        if args.round is not None:
            load[not_at_columns] = load[not_at_columns].round(args.round)
        load.insert(0, 'task', [fname.lstrip('_').split('_',1)[0]+' '+fname.rsplit('_',1)[1].split('.',1)[0]])
        if data is None:
            data = load
        else:
            data = pd.concat((data,load))
    return data

def main(args=None, prs=None):
    if prs is None:
        prs = build()
    args = parse(prs, args)
    if args.subparser_name == 'task':
        data = load_task_inputs(args)
        if args.output_name is not None and os.path.exists(args.output_name) and not args.overwrite:
            print(f"WARNING! {args.output_name} already exists and would be overwritten!")
            print("Rerun this script with --overwrite if it is ok to replace this file, or move it to a different path")
            exit()
        table_analyze(data, args)
    elif args.subparser_name == 'collate':
        data = load_collate_inputs(args)
        if args.latex:
            latexify(data,args)
        else:
            print(data.to_string(index=False))

if __name__ == '__main__':
    main()

