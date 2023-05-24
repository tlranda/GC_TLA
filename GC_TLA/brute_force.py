import argparse
import os
import numpy as np, pandas as pd

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--exhaust", required=True, help="Exhaustive data to utilize")
    prs.add_argument("--traces", required=True, nargs="+", help="Experimental traces to rank")
    prs.add_argument("--as-percentile", action="store_true", help="Show percentiles instead of absolute ranks")
    prs.add_argument("--show-all", action="store_true", help="Display all ranks instead of just summaries")
    prs.add_argument("--round", type=int, default=None, help="Round to this many places (default: No rounding)")
    prs.add_argument("--plot", action="store_true", help="Produce a plot representation when specified")
    prs.add_argument("--save-name", type=str, default="BruteForce", help="Plot saved to this name (default %(default)s)")
    prs.add_argument("--format", choices=['svg','png','pdf'], default='svg', help="Plot format when saving (default %(default)s)")
    prs.add_argument("--zoomed", action='store_true', help="Include zoom to top-10 evaluations")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # File checks
    not_found = []
    if not os.path.exists(args.exhaust):
        not_found.append(args.exhaust)
    for fname in args.traces:
        if not os.path.exists(fname):
            not_found.append(fname)
    if len(not_found) > 0:
        raise ValueError(f"Unable to find file(s): {', '.join(not_found)}")
    return args

def load(args):
    # Loaded astype str for exact match lookup semantics used later
    exhaust = pd.read_csv(args.exhaust).sort_values(by='objective').drop(['predicted','elapsed_sec'],axis=1)
    # Push objective == 1.0 (sentinel failure value) to the very end
    # Also make it flatline up there
    movement = np.where(exhaust['objective'] == 1.0, True, False)
    exhaust = exhaust.reindex(exhaust.index[~movement].tolist() + exhaust.index[movement].tolist()).reset_index(drop=True)
    exhaust.loc[(len(exhaust)-1-np.arange(sum(movement))).tolist(), 'objective'] = np.max(exhaust['objective'])
    exhaust = exhaust.astype(str)
    traces = []
    for fname in args.traces:
        traces.append(pd.read_csv(fname).astype(str))
    return exhaust, traces

def find_exhaust_row(exhaust, row, cand_cols):
    # The row that matches the count of columns is a full match -- ie the rank of the given row parameterization
    search_tup = tuple(row[list(cand_cols)].values)
    n_matching = (exhaust[list(cand_cols)] == search_tup).sum(1)
    matches = np.where(n_matching == len(cand_cols))[0]
    return matches[0]
    # We stop above to reduce the needed access count, but more exhaustive form below.
    # We reordered exhaustive data before searching so that these operations are unnecessary.
    #
    # match_data = exhaust.iloc[matches]
    # rowid = match_data.index[0]
    # return rowid

def reidentify(exhaust, traces):
    reidentified = []
    cand_cols = tuple([_ for _ in traces[0].columns if _ != 'objective' and _ in exhaust.columns])
    for trace in traces:
        ids = []
        for (idx, row) in trace.iterrows():
            ids.append(find_exhaust_row(exhaust, row, cand_cols))
        reidentified.append(ids)
    return reidentified

def present(data, exhaust, args):
    # Show the exhaustive ranks
    maxrank = len(exhaust)
    if args.as_percentile:
        print(f"Out of 100% (lower is better rank)...")
    else:
        print(f"Out of {maxrank} possible configurations (lower is better rank)...")
    for fname, fdata in zip(args.traces, data):
        # Transform into percentages if requested
        if args.as_percentile:
            fdata = np.asarray(fdata)/maxrank*100
        if args.round is not None:
            fdata = np.round(np.asarray(fdata), args.round)
        print(f"{fname}:")
        print("\t"+f"Best|Avg|Worst Rank: {np.min(fdata)}|",end="")
        if args.round is not None:
            print(f"{np.round(np.mean(fdata), args.round)}|", end='')
        else:
            print(f"{np.mean(fdata)}|", end='')
        print(f"{np.max(fdata)}")
        if args.show_all:
            print("\t"+fdata)

replaceable = {'jaehoon': 'BO', 'thomas': 'GC', 'gptune': 'GPTune'}
def nicename(name):
    name = name.rsplit('/',1)[1]
    if name.endswith('_experiments'):
        name = name[:-12]
    if name in replaceable.keys():
        name = replaceable[name]
    return name

def plot(data, exhaust, args):
    import matplotlib
    # Adjustments
    font = {'size': 14, 'family': 'serif'}
    lines = {'linewidth': 3, 'markersize': 6}
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
    fig, ax = plt.subplots()
    # Have to reconvert back to float
    exhaust['objective'] = exhaust['objective'].astype(float)
    # Start with objective
    ax = exhaust['objective'].plot(ax=ax,legend=False, label="Objective",zorder=-1)
    # Recombine based on directory and nicename them
    to_plot = {}
    for src, d in zip(args.traces, data):
        dirname = nicename(os.path.dirname(src))
        if dirname in to_plot.keys():
            to_plot[dirname].append(d)
        else:
            to_plot[dirname] = [d]
    # Add data points gradually
    colors = ['tab:red','tab:orange','tab:olive']
    markersize=125
    alpha=1
    for color, (plot_name, plot_data) in zip(colors, to_plot.items()):
        arr = np.atleast_2d(plot_data)
        # Best
        best = arr.ravel()[np.argmin(arr)]
        if not args.zoomed and plot_name == 'GC':
            ax.scatter(best, exhaust['objective'].iloc[best], marker=matplotlib.markers.MarkerStyle('v',fillstyle='left'), color=color, label=plot_name+" Best", s=markersize, alpha=alpha)
        elif not args.zoomed and plot_name == 'GPTune':
            ax.scatter(best, exhaust['objective'].iloc[best], marker=matplotlib.markers.MarkerStyle('v',fillstyle='right'), color=color, label=plot_name+" Best", s=markersize, alpha=alpha)
        else:
            ax.scatter(best, exhaust['objective'].iloc[best], marker='v', color=color, label=plot_name+" Best", s=markersize, alpha=alpha)
        # Worst
        #worst = arr.ravel()[np.argmax(arr)]
        #ax.scatter(worst, exhaust['objective'].iloc[worst], marker='X', color=color, label=plot_name+" Worst", s=markersize, alpha=alpha)
        # Average
        avg = int(np.mean(arr))
        ax.scatter(avg, exhaust['objective'].iloc[avg], marker='o', color=color, label=plot_name+" Average", s=markersize, alpha=alpha)
    # Add zoomed-in inset
    if args.zoomed:
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
        zoomed_in = zoomed_inset_axes(plt.gca(), 500, loc='center')
        zoomed_in = exhaust['objective'].plot(ax=zoomed_in,legend=False, label="Objective",zorder=-1)
        for color, (plot_name, plot_data) in zip(colors, to_plot.items()):
            arr = np.atleast_2d(plot_data)
            # Best
            best = arr.ravel()[np.argmin(arr)]
            zoomed_in.scatter(best, exhaust['objective'].iloc[best], marker='v', color=color, label=plot_name+" Best", s=markersize, alpha=alpha)
            # Worst
            #worst = arr.ravel()[np.argmax(arr)]
            #zoomed_in.scatter(worst, exhaust['objective'].iloc[worst], marker='X', color=color, label=plot_name+" Worst", s=markersize, alpha=alpha)
            # Average
            avg = int(np.mean(arr))
            zoomed_in.scatter(avg, exhaust['objective'].iloc[avg], marker='o', color=color, label=plot_name+" Average", s=markersize, alpha=alpha)
        zmin, zmax = 0, 10
        zoomed_in.set_xlim(zmin,zmax)
        zoomed_in.set_ylim(exhaust['objective'].iloc[zmin], exhaust['objective'].iloc[zmax])
        mark_inset(ax, zoomed_in, loc1=2, loc2=4, fc='none', ec='0.5')
    # Generic plot stuff
    ax.set_ylabel("Objective Time (seconds)")
    #ax.set_yscale('log')
    ax.set_ylim(0,20)
    #ax.set_xscale('log')
    ax.set_xlabel("Configuration")
    #ax.set_title("Brute Force Syr2k")
    # Custom legends
    locs = ['upper left',
            'upper center',
            'center left']
    markers = ['v',
               #'X',
               'o']
    names = ['Best',
             #'Worst',
             'Average']
    for key, loc, color in zip(to_plot.keys(), locs, colors):
        legend_elems = [matplotlib.lines.Line2D([0],[0], marker=m, markeredgecolor=color, markerfacecolor=color, label=key+' '+kind, markersize=10) for m,kind in zip(markers,names)]
        if loc.startswith('upper'):
            kwargs = {}
        else:
            kwargs = {'bbox_to_anchor': (0,0.7)}
        new_leg = ax.legend(handles=legend_elems, loc=loc, **kwargs)
        ax.add_artist(new_leg)
    #legend_elems = [matplotlib.patches.Patch(facecolor=color, label=key) for color, key in zip(colors, to_plot.keys())]
    #legend_elems += [matplotlib.lines.Line2D([0],[0], marker=m, markerfacecolor='k', label=kind, markersize=15) for m,kind in zip(['v','X','o'],['Best','Worst','Average'])]
    #ax.legend(handles=legend_elems, loc='upper left')
    fig.tight_layout()
    plt.savefig(args.save_name+'.'+args.format, format=args.format)

def main(args=None, prs=None):
    if prs is None:
        prs = build()
    args = parse(prs, args)
    exhaust, traces = load(args)
    reidentified = reidentify(exhaust, traces)
    present(reidentified, exhaust, args)
    if args.plot:
        plot(reidentified, exhaust, args)

if __name__ == '__main__':
    main()
