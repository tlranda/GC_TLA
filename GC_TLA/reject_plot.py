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
import matplotlib.pyplot as plt
rcparams = {'axes.labelsize': 14,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            }
plt.rcParams.update(rcparams)
import argparse, inspect, re

# Fetch plottable methods for use as arguments and their reference to method object
def get_methods():
    ignore_methods = ['parse', 'main', 'load', 'finalize']
    methods = dict((k,v) for (k,v) in globals().items() if k not in ignore_methods and callable(v) and 'args' in inspect.signature(v).parameters)
    return methods

def build():
    methods = get_methods()
    prs = argparse.ArgumentParser()
    prs.add_argument('--files', required=True, nargs='+', type=str, help="Files to load")
    prs.add_argument('--ignore', default=None, nargs="+", type=str, help="Globbed paths to ignore")
    prs.add_argument('--call', required=True, nargs='+', type=str, choices=list(methods.keys()), help="Methods to call")
    prs.add_argument('--name', type=str, help="Name suggestion for generated image files")
    prs.add_argument('--space', type=float, default=0, help="Horizontal space between bars")
    prs.add_argument('--xlim', type=float, default=None, help="Limit max x range")
    prs.add_argument('--reject-only', action='store_true', help="Do not include accepted as a configuration type for '--call reject'")
    prs.add_argument("--fig-dims", metavar=("Xinches", "Yinches"), nargs=2, type=float,
                     default=plt.rcParams["figure.figsize"], help="Figure size in inches "
                     f"(default is {plt.rcParams['figure.figsize']})")
    prs.add_argument("--fig-pts", type=float, default=None, help="Specify figure size using LaTeX points and Golden Ratio")
    prs.add_argument("--format", choices=["png", "pdf", "svg","jpeg"], default="pdf", help="Format to save outputs in")
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
    if args.fig_pts is not None:
        args.fig_dims = set_size(args.fig_pts)
    if args.ignore is not None:
        new_files = []
        for file in args.files:
            if file not in args.ignore:
                new_files.append(file)
        args.files = new_files
    print(args.files)
    return args

# Data loading here is just a dictionary of names: CSV representations
def load(args):
    return dict((k,pd.read_csv(k)) for k in args.files)

# Make a cleaner filename for legends etc
def name_cleaner(name):
    name = os.path.basename(name)
    # Expected format: REJECT_{METHOD}_problem.{SIZE}_{SEED}.csv
    fields = name.split('_')
    fields[2] = fields[2].split('.')[1]
    name = '_'.join([fields[_] for _ in [1,2]])
    return name

def mpl_name(li):
    for name in li:
        yield re.sub(r"([A-Z]*)([a-z])([A-Z])",r"\1\2\n\3", name[0].upper()+name[1:])

# Plot time spent sampling (ONLY SAMPLING) vs #sampling iterations
def iter_time(data, args):
    info = {}
    fig, ax = plt.subplots(figsize=tuple(args.fig_dims))
    fig.set_tight_layout(True)
    ys, names = [],[]
    arbitrary_order = ['random', 'GaussianCopula', 'CTGAN', 'CopulaGAN']
    data_order = []
    for name in data.keys():
        set_idx = -1
        for idx, key in enumerate(arbitrary_order):
            if key in name:
                data_order.append(idx)
                set_idx = idx
                break
        if set_idx < 0:
            data_order.append(len(arbitrary_order))
    key_order = np.asarray(list(data.keys()))[np.argsort(data_order)]
    for name in key_order:
        sequence = data[name]
        nicename = name_cleaner(name)
        try:
            ys.append(max(sequence['sample.1']))
        except ValueError:
            ys.append(np.inf)
        names.append(nicename.rsplit('_',1)[0])
        #line2 = ax.plot(1+sequence.trial, sequence['sample.1']+sequence['external'], marker=',', linestyle='--', color=line1[0].get_color())
    maxheight = np.asarray(ys)[np.isfinite(ys)].max()
    heightsort = np.argsort(ys)
    for leftToRight, idx in enumerate(heightsort):
        nicename, height = names[idx], ys[idx]
        line1 = ax.bar(leftToRight, min(height, maxheight*10), label=f"{nicename}")
        print(f"Plot {nicename} at height {min(height, maxheight*2)}")
    info['min_x'] = 1
    info['pre_legend'] = True
    ax.set_yscale('log')
    ax.set_ylim([None, 10**int(round(np.log10(maxheight),0))])
    ax.set_ylabel('Time to Generate 1000 Samples (seconds)')
    ax.set_xticks(range(len(names)))
    sort_names = np.asarray([_ for _ in mpl_name(names)])[heightsort]
    ax.set_xticklabels(sort_names)
    return f'iter_time.{args.format}', info, fig, ax

# Plot #accepted samples vs #sampling iterations [[DROPPED FIGURE CONCEPT: Nondescriptive for things worth talking about]]
def generate(data, args):
    info = {}
    fig, ax = plt.subplots(figsize=tuple(args.fig_dims))
    fig.set_tight_layout=True
    max_len = 0
    for name, sequence in data.items():
        nicename = name_cleaner(name)
        line = ax.plot(1+sequence.trial, sequence['generate'], marker=',', label=f"{nicename} 1000 Samples")
        if len(sequence.trial) > 0:
            max_len = max(max_len, max(1+sequence.trial))
    ax.plot([_ for _ in range(1,max_len+1)], [1000 for _ in range(max_len)], linestyle='--')
    info['min_x'] = 1
    ax.set_ylabel('# Accepted Configurations')
    ax.set_xlabel('# Sampling Iterations')
    return f'generate.{args.format}', info, fig, ax

# Plot #rejected samples (by reason) vs #sampling iterations
def reject(data, args):
    info = {}
    fig, ax = plt.subplots(figsize=tuple(args.fig_dims))
    fig.set_tight_layout=True
    nbars = len(list(data.keys()))
    max_len = 0
    barlookups = {'Accepted': ['generate'],
                  'Repeated Configuration': ['sample', 'batch', 'prior'],
                  'Ill-Conditioned': ['close'],
                 }
    if args.reject_only:
        del barlookups['Accepted']
    hatches = [None, 'OO', 'XX']
    #barkeys = ['close','sample', 'batch', 'prior']
    #hatches = [None, 'XX', '--', 'OO']
    arbitrary_order = ['random', 'GaussianCopula', 'CTGAN', 'CopulaGAN']
    data_order = []
    for name in data.keys():
        set_idx = -1
        for idx, key in enumerate(arbitrary_order):
            if key in name:
                data_order.append(idx)
                set_idx = idx
                break
        if set_idx < 0:
            data_order.append(len(arbitrary_order))
    key_order = np.asarray(list(data.keys()))[np.argsort(data_order)]
    for idx, name in enumerate(key_order):
        sequence = data[name]
        print(name)
        print(sequence)
        nicename = name_cleaner(name)
        nicename, SIZE = nicename.split('_')
        bottom = [0 for _ in sequence.trial]
        if len(sequence.trial) > 0:
            max_len = max(max_len, max(1+sequence.trial))
        color = None
        #for key, hatch in zip(barkeys, hatches):
        for ((key, keylist), hatch) in zip(barlookups.items(), hatches):
            # Imaginary sequence for legend
            if color is None:
                series_color = ax.bar([1],[1], zorder=-1, label=nicename, edgecolor='black', width=1/8)
                color = series_color.patches[0].get_facecolor()
            sequence_stack = sequence[keylist[0]]
            for seq in keylist[1:]:
                sequence_stack += sequence[seq]
            # DON'T PLOT CUMULATIVELY -- PLOT DIFFERENCES
            if len(sequence_stack) > 1:
                sequence_stack = [sequence_stack[0]]+[j-i for (i,j) in zip(sequence_stack[:-1], sequence_stack[1:])]
            bar = ax.bar(1+idx+((nbars+args.space)*sequence.trial), sequence_stack, bottom=bottom, color=color, hatch=hatch, edgecolor='black')
            if len(sequence.trial) > 0:
                bottom = np.asarray(sequence_stack) + bottom
    info['min_x'] = 0
    # This legend gets deleted so add it back afterwards
    l1 = ax.legend(loc='lower left') # All of the line infos
    # Hatch infos
    bars = []
    for key, hatch in zip(barlookups.keys(), hatches):
        bars.append(matplotlib.patches.Patch(facecolor='white', edgecolor='black', label=key, hatch=hatch))
    # Put order backwards to match stack order in the plot
    bars.reverse()
    l2 = ax.legend(handles=bars, loc='lower right')
    ax.add_artist(l1) # Slap it back on there
    info['pre_legend'] = True
    ax.set_yscale('log')
    if args.reject_only:
        ax.set_ylabel('# Rejected Configurations')
    else:
        ax.set_ylabel('# Configurations')
    ax.set_xlabel('# Sampling Iterations')
    ax.set_xticks([1+nbars//2+((nbars+args.space)*_) for _ in range(max_len)], [_ for _ in range(1,max_len+1)])
    ax.set_title(f'Generated Configurations for Syr2k {SIZE}')
    return f'reject.{args.format}', info, fig, ax

# CSV FORMAT
# trial, generate, reject, close, sample, batch, prior, sample.1, external

# Common operations to finalize the figure
def finalize(names, infos, figures, axes, args):
    for name, info, fig, ax in zip(names, infos, figures, axes):
        # INFO used to identify if legend applied or not
        if 'pre_legend' not in info.keys() or not info['pre_legend']:
            ax.legend()
        # ZOOM
        if args.xlim is not None:
            ax.set_xlim([info['min_x'], args.xlim])
        # SAVE IMAGE FILE
        name = name.rsplit('.',1)
        if args.name is not None and args.name != '':
            name[0] += args.name
        name = '.'.join(name)
        fig.savefig(name, format=args.format, bbox_inches='tight')

def main(args=None):
    if args is None:
        args = parse(build())
    # Load once, drive calls while collating returns, then mass-finalize
    methods = get_methods()
    data = load(args)
    names, info, figures, axes = [], [], [], []
    for call in args.call:
        n, i, f, a = methods[call](data, args)
        names.append(n)
        info.append(i)
        figures.append(f)
        axes.append(a)
    finalize(names, info, figures, axes, args)

if __name__ == '__main__':
    main()

