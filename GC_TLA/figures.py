import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# Get legend names from matplotlib
from matplotlib.offsetbox import AnchoredOffsetbox
legend_codes = list(AnchoredOffsetbox.codes.keys())+['best']
import argparse
import os

rfigures = {4: ['anal_density', '4', 'time_v_exec'],
            5: ['plot_best_sofar', '5', 'elapse_v_best_speedup'],
            6: ['plot_time', '6', 'elapse_v_no_configs_speedup'],
            7: ['run_times', '7', 'no_evals_v_exec'],
            1: ['rank_trace','1','rank_vs_eval']
           }
argfigures = {}
for (k,v) in rfigures.items():
    for vv in v:
        argfigures[vv] = k

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--figure', choices=list(argfigures.keys()), nargs='+', required=True, help="Figure(s) to plot")
    prs.add_argument('--data', type=str, help='Data directory to load from')
    prs.add_argument('--files', type=str, nargs='+', required=True, help="Files to read")
    prs.add_argument('--legend', choices=legend_codes, default=None, help="Legend location; note two-word legends must be quoted on command line")
    prs.add_argument("--log-x", action="store_true", help="Logarithmic x axis")
    prs.add_argument("--log-y", action="store_true", help="Logarithmic y axis")
    prs.add_argument('--alpha', type=float, default=0.1, help='Cutoff value quantile')
    prs.add_argument('--n-infer', type=int, default=200, help='Cutoff number of points')
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    args.figure = [argfigures[f] for f in args.figure]
    if len(args.data) > 0 and not args.data.endswith('/'):
        args.data += '/'
    new_names = []
    for f in args.files:
        if f.startswith(args.data):
            f = f[len(args.data):]
        new_names.append(f)
    args.files = new_names
    return args

SIZES = ['s', 'sm', 'm', 'ml', 'l', 'xl']
SOURCES = [_ for _ in SIZES if len(_) == 1]
TARGETS = [_ for _ in SIZES if _ not in SOURCES]

def labeler(fname):
    if 'bootstrap' in fname.lower():
        return 'Bootstrap'
    elif '_top_' in fname.lower():
        return fname
    elif 'no_refit' in fname.lower():
        return 'SDV'
    elif 'refit_' in fname.lower():
        return 'SDV_Refit'
    elif 'gptune' in fname.lower():
        return 'GPTune'
    else:
        return fname

def load_o3s(args):
    # Get -O3 times as available
    o3 = dict()
    for s in SIZES:
        try:
            time = pd.read_csv(f"{args.data}DEFAULT_{s.upper()}.csv")
            o3[s] = float(time['objective'].iloc[0])
        except:
            pass
    loaded_sizes = list(o3.keys())
    return o3, loaded_sizes


def fig4(args):
    for target in TARGETS:
        fig, ax = plt.subplots()
        for fname in args.files:
            if target.lower() not in fname and target.upper() not in fname:
                continue
            csv = pd.read_csv(f"{args.data}{fname}")
            obj_col = 'objective' if 'objective' in csv.columns else 'exe_time'
            lname = labeler(os.path.basename(fname))
            ax.plot(csv['elapsed_sec'], csv[obj_col], '.-', label=lname,) #color=""
        ax.set_ylabel('Execution time', fontsize=14)
        ax.set_xlabel('Time (s.)', fontsize=14)
        ax.set_xlim(0)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12, loc=args.legend)
        ax.grid()
        plt.savefig(f"fig4_{target}.png")

def fig5(args):
    o3, loaded_sizes = load_o3s(args)
    gathered, cutoffs = [], []
    for target in TARGETS:
        if target not in loaded_sizes:
            print(f"Warning! Size {target} had no -O3 value, skipping.")
            continue
        o3_time = o3[target]
        evals = []
        info = {}
        for fname in args.files:
            if target.lower() not in fname and target.upper() not in fname:
                continue
            csv = pd.read_csv(f"{args.data}{fname}")
            kernel_evals = (o3_time / np.array(csv['objective'])).tolist()
            evals.extend(kernel_evals)
            lname = labeler(os.path.basename(fname))
            if lname in info.keys():
                info[lname]['evals'].extend(kernel_evals)
                info[lname]['t'] = np.hstack((info[lname]['t'], np.array(csv['elapsed_sec'])))
                #info[lname]['t'].extend(np.array(csv['elapsed_sec']))
            else:
                info[lname] = {'evals': kernel_evals,
                               'counter': 0,
                               'plot': [0],
                               'best': 0.,
                               'idx': 0,
                               't': np.array(csv['elapsed_sec']),
                              }
        if len(evals) == 0:
            continue
        cutoff = sorted(evals, reverse=True)[int(len(evals)*args.alpha)]
        print(f"{target} cutoff: {cutoff}")

        t = 0
        T_MAX = int(max([info[k]['t'].max() for k in info.keys()]))
        while t < T_MAX:
            for k in info.keys():
                idx = info[k]['idx']
                try:
                    if info[k]['t'][idx] < t and idx < args.n_infer:
                        tmp = float(info[k]['evals'][idx])
                        info[k]['best'] = max(info[k]['best'], tmp)
                        info[k]['plot'].append(info[k]['best'])
                        info[k]['idx'] += 1
                    elif idx == args.n_infer:
                        pass
                    else:
                        info[k]['plot'].append(info[k]['best'])
                except IndexError:
                    pass
            t += 1
        gathered.append(dict((k, info[k]['plot']) for k in info.keys()))
    fig, axs = plt.subplots(1,3, figsize=(15,3), sharex=False, sharey=True)
    for ax, cc, target in zip(axs.flat, gathered, TARGETS):
        for (k,v) in cc.items():
            ax.plot(range(1,len(v)+1), v, '.-', label=k, linewidth=1.5, markersize=1.5,)# color=""
            ax.legend(fontsize=13, loc=args.legend)
            ax.set_ylabel('Best speedup so far', fontsize=12)
            ax.set_xlabel('Elapsed time (Seconds)', fontsize=12)
            #ax.set_title(kernel+"_"+target, fontsize=12)
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=12)
            ax.grid()
    plt.savefig(f"fig5.png")

def fig6(args):
    o3, loaded_sizes = load_o3s(args)
    gathered, cutoffs = [], []
    for target in TARGETS:
        if target not in loaded_sizes:
            print(f"Warning! Size {target} had no -O3 value, skipping.")
            continue
        o3_time = o3[target]
        evals = []
        info = {}
        for fname in args.files:
            if target.lower() not in fname and target.upper() not in fname:
                continue
            csv = pd.read_csv(f"{args.data}{fname}")
            kernel_evals = o3_time / np.array(csv['objective'])
            evals.extend(kernel_evals)
            lname = labeler(os.path.basename(fname))
            if lname in info.keys():
                info[lname]['evals'].extend(kernel_evals)
                info[lname]['t'].extend(np.array(csv['elapsed_sec']))
            else:
                info[lname] = {'evals': kernel_evals,
                               'counter': 0,
                               'plot': [0],
                               'best': 0.,
                               'idx': 0,
                               't': np.array(csv['elapsed_sec']),
                              }
        cutoff = sorted(evals, reverse=True)[int(len(evals)*args.alpha)]
        print(f"{target} cutoff: {cutoff}")

        t = 0
        T_MAX = int(max([info[k]['t'].max() for k in info.keys()]))
        while t < T_MAX:
            for k in info.keys():
                idx = info[k]['idx']
                try:
                    if info[k]['t'][idx] < t and idx < args.n_infer:
                        tmp = info[k]['evals'][idx]
                        if str(tmp) != 'nan':
                            if tmp > cutoff:
                                info[k]['counter'] += 1
                            info[k]['plot'].append(info[k]['counter'])
                        info[k]['idx'] += 1
                    else:
                        info[k]['plot'].append(info[k]['counter'])
                except IndexError:
                    pass
            t += 1
        gathered.append(dict((k, info[k]['plot']) for k in info.keys()))
        cutoffs.append(cutoff)
    fig, axs = plt.subplots(1,3, figsize=(15,3), sharex=False, sharey=True)
    for ax, cc, d_size, cutoff in zip(axs.flat, gathered, loaded_sizes, cutoffs):
        for k,v in cc.items():
            ax.plot(range(1, len(v)+1), v, '.-', label=k, linewidth=1.5, markersize=1.5,) #color=color
        ax.legend(fontsize=13)
        ax.set_ylabel(f'No. of config speedup > top {args.alpha*100}% = {str(cutoff)[:4]}', fontsize=12)
        ax.set_xlabel('Elapsed time (seconds)', fontsize=12)
        #ax.set_title(f'{kernel}_{d_size}', fontsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid()
    plt.savefig(f"fig6.png")

def fig7(args):
    gathered = []
    for target in TARGETS:
        info = {}
        for fname in args.files:
            if target.lower() not in fname and target.upper() not in fname:
                continue
            csv = pd.read_csv(f"{args.data}{fname}")
            info[labeler(os.path.basename(fname))] = np.asarray(csv['objective'])
        gathered.append(info)
    fig, axs = plt.subplots(3,1, figsize=(15,12), sharex=True, sharey=False)
    for ax, objectives in zip(axs.flat, gathered):
        for k,v in objectives.items():
            ax.plot(range(1, len(v)+1), v, '.-', label=k, linewidth=0.5, markersize=3,) #color=color
        ax.legend(fontsize=13)
        ax.set_ylabel('Execution time (sec.)', fontsize=12)
        ax.set_xlabel('No. of Evaluations', fontsize=12)
        #ax.set_title(f'{kernel}_{d_size}', fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid()
    plt.savefig(f"fig7.png")

def fig1(args):
    gathered = {}
    for fname in args.files:
        csv = pd.read_csv(f"{args.data}{fname}")
        label = os.path.basename(fname)
        try:
            nlabel = label[label.index('REFIT')+6:]
            nlabel = nlabel[:nlabel.index('_')]
            label = f'REFIT {int(nlabel)}'
        except ValueError:
            label = 'NO REFIT'
        gathered[label] = csv['objective'].argsort()
    LIMIT = 40
    fig, ax = plt.subplots()
    for k,v in gathered.items():
        min_sofar = [v[0]]
        for limiter, item in zip(range(LIMIT-1), v[1:]):
            min_sofar.append(min(min_sofar[-1], item))
        #ax.plot([_ for _ in range(len(v))], v, label=k, linewidth=0.5, markersize=3,)
        ax.plot([_ for _ in range(min(LIMIT,len(v)))], min_sofar, label=k, linewidth=0.5, markersize=3,)
    ax.legend(fontsize=13)
    ax.set_ylabel('Evaluation Rank')
    ax.set_ylim([0,10])#int(0.1*len(v))])
    ax.set_xlabel('Evaluation')
    #ax.set_title()
    ax.tick_params(axis='x',labelsize=12)
    ax.tick_params(axis='y',labelsize=12)
    ax.grid()
    plt.savefig(f"fig1.png")

def main(args):
    fig_methods = dict((int(k[3:]),v) for (k,v) in globals().items() if k.startswith('fig') and callable(v))
    for figure in args.figure:
        fig_methods[figure](args)

if __name__ == '__main__':
    main(parse(build()))

