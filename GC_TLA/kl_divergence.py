import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import argparse, os
from scipy.stats import entropy

def build():
    prs = argparse.ArgumentParser()
    data_args = prs.add_argument_group("Data", "Arguments that affect data intake and processing")
    data_args.add_argument('--exhaust', metavar='ex.csv', type=str, required=True,
                            help="Exhaustive file for 'truth' distribution")
    data_args.add_argument('--x-ratio', metavar='x%', dest='x_k', type=float, default=1.0, nargs='+',
                            help="Limit exhaustive to top-%% evaluations as ratio (default: %(default)s)")
    data_args.add_argument('--sample', metavar='*.csv', type=str, required=True, nargs='+',
                            help="Sample file(s) for 'predict' distribution (collated if multiple)")
    data_args.add_argument('--s-ratio', metavar='s%', dest='p_k', type=float, default=1.0, nargs='+',
                            help="Limit samples to top-%% evaluations (per file) as ratio (default: %(default)s)")

    plot_args = prs.add_argument_group("Plotting", "Arguments that affect plot labels, sizes, and outputs")
    plot_args.add_argument('--version', choices=['1','2'], default='2',
                            help='Iteration of plot design to use (default %(default)s)')
    plot_args.add_argument('--expand-x', metavar='FACTOR', type=float, default=1,
                            help="Factor to adjust figsize in x-dimension (default: %(default)s)")
    plot_args.add_argument('--too-long', metavar='N', type=int, default=15,
                            help="Maximum name length (default: %(default)s)")
    plot_args.add_argument('--abbrev', metavar='M', type=int, default=5,
                            help="When shortening names, keep M prefix and M postfix characters in abbreviation (default: %(default)s)")
    plot_args.add_argument('--save-name', metavar='NAME', type=str, default=None,
                            help="Save generated figure to NAME (if not given, display immediately)")
    plot_args.add_argument('--format', type=str, choices=['svg','png','pdf'], default='svg',
                            help="Format type for Matplotlib to save the figure as (default: %(default)s)")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    args.version = int(args.version)
    if args.sample is None:
        args.sample = []
    for metavar in [arg for arg in dir(args) if arg.endswith('_k')]:
        readattr = getattr(args,metavar)
        # Enforce list type
        if type(readattr) is not list:
            readattr = [readattr]
        # Sort list
        if args.version == 1:
            setattr(args,metavar,sorted(readattr, reverse=True))
        elif args.version == 2:
            setattr(args,metavar,sorted(readattr))
        else:
            setattr(args,metavar,readattr)
    return args

def load_files(args):
    # Load files and apply topK filtering if needed
    # DO NOT trim to top-% exhaustive yet, as full data is needed for proper functionality
    exhaust = pd.read_csv(args.exhaust).sort_values(by='objective').reset_index(drop=True)
    sample = [pd.read_csv(s).sort_values(by='objective').reset_index(drop=True) for s in args.sample]
    # Collate samples per value of p_k
    collate = []
    for p_k in args.p_k:
        collate.append(pd.concat([s.iloc[:int(p_k*len(s))] for s in sample]))
    return exhaust, collate

def make_dist(value_dict, data):
    # Use value dictionary to get distribution histogram for this dataset
    breakdown = {}
    common_denom = len(data)
    for key, values in value_dict.items():
        keydata = list(data[key])
        breakdown[key] = [keydata.count(val) / common_denom for val in values]
    return breakdown

def kl_div_per_sample(fig, ax, exhaust, sampled, concentration, args):
    cols = sorted([_ for _ in exhaust.columns if _.startswith('p') and _ != 'predicted'])
    value_dict = dict((k, sorted(set(exhaust[k]))) for k in cols)
    # Do KL-Divergence per variable trace
    kl_div = np.zeros((len(cols),len(args.x_k)))
    sampled_dist = make_dist(value_dict, sampled)
    for y, exhaust_concentration in enumerate(args.x_k):
        exhaust_dist = make_dist(value_dict, exhaust.iloc[:int(exhaust_concentration*len(exhaust))])
        for x, col in enumerate(cols):
            kl_div[x,y] = entropy(exhaust_dist[col], sampled_dist[col])
            if not np.isfinite(kl_div[x,y]):
                # Usually due to sampled dist having 0 probability too much; swapping distributions order can measure
                # NOTE: KL Divergence is a metric, but this swap is NOT equivalent and MAY be bad... but gives more information than INF
                kl_div[x,y] = entropy(sampled_dist[col], exhaust_dist[col])
    # Plot as separate lines
    for x, col in enumerate(cols):
        if kl_div.shape[1] > 1:
            ax.plot(args.x_k, kl_div[x,:], label=col, marker='.')
        else:
            ax.bar(x, kl_div[x,:], label=col)
    # Common plot transformations
    ax.set_title(f"Top {concentration} Sample Data")
    ax.set_xlabel("Exhaustive Quantile")
    if concentration >= 1.0:
        ax.set_ylabel("KL Divergence")
        ax.legend()
    #if args.save_name is None:
    #    plt.show()
    #else:
    #    fig.savefig(args.save_name+f'_{concentration}', format='png')

singleton_titles = ["Random", "Highest-Performing"]
def kl_div_per_concentration(fig, ax, exhaust, samples, concentration, args):
    cols = sorted([_ for _ in exhaust.columns if _.startswith('p') and _ != 'predicted'])
    value_dict = dict((k, sorted(set(exhaust[k]))) for k in cols)
    kl_div = np.zeros((len(args.p_k), len(cols)))
    exhaust_dist = make_dist(value_dict, exhaust.iloc[:int(concentration*len(exhaust))])
    space_sizes = []
    for x, sampled_dist in enumerate([make_dist(value_dict, sampled) for sampled in samples]):
        space_sizes.append(np.prod([(np.asarray(sampled_dist[f'p{i}'])>0).sum() for i in range(6)]))
        for y, col in enumerate(cols):
            kl_div[x,y] = entropy(exhaust_dist[col], sampled_dist[col])
            if not np.isfinite(kl_div[x,y]):
                # Usually due to 0 probability too much in a distribution
                kl_div[x,y] = entropy(sampled_dist[col], exhaust_dist[col])
    # Plot as ONE LINE
    ax.plot(args.p_k, np.mean(kl_div, axis=1), marker='.')
    title = singleton_titles.pop()
    print(title)
    print(args.p_k)
    print(np.mean(kl_div,axis=1))
    print(space_sizes)
    # Common plot transformations
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel("Fit Quantile")
    if concentration == args.x_k[0]:
        ax.set_ylabel("KL Divergence")

def main(args=None):
    if args is None:
        args = parse(build())
    exhaust, samples = load_files(args)
    default_figsize = plt.rcParams['figure.figsize']
    if args.version == 1:
        fig, axes = plt.subplots(1, len(args.p_k), sharey=True, figsize=(default_figsize[0]*args.expand_x, default_figsize[1]))
        for sample_concentration, sample_frame, ax in zip(args.p_k, samples, axes):
            kl_div_per_sample(fig, ax, exhaust, sample_frame, sample_concentration, args)
    elif args.version == 2:
        fig, axes = plt.subplots(1,2, sharey=True, figsize=(default_figsize[0]*args.expand_x, default_figsize[1]))
        for exhaust_concentration, ax in zip(args.x_k, axes):
            kl_div_per_concentration(fig, ax, exhaust, samples, exhaust_concentration, args)
    fig.tight_layout()
    if args.save_name is None:
        plt.show()
    else:
        fig.savefig(args.save_name, format=args.format)

if __name__ == '__main__':
    main()

