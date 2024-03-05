import os
import importlib, skopt, argparse, matplotlib
font = {'size': 14,
        'family': 'serif',
        }
lines = {'linewidth': 3,
         'markersize': 6,
        }
matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)
import pandas as pd, numpy as np, matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = "/usr/bin/ffmpeg"
rcparams = {'axes.labelsize': 14,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            }
plt.rcParams.update(rcparams)
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from sklearn import neighbors

# From syr2k_exp dir:
# python ../tsne_figure.py --problem syr2k_exp.problem.S --convert data/results_rf*.csv data/thomas_experiments/syr2k_NO_REFIT_GaussianCopula_*_1234* --quantile 0.3 0.3 0.3 1 1 1 --rank-color

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--problem", required=True, help="Problem to reference for params (as module import)")
    prs.add_argument("--convert", nargs="+", required=True, help="Files to collate/convert")
    prs.add_argument("--quantile", nargs="+", required=True, type=float, help="Quantiles PER FILE or GLOBAL to highlight 'optimum' portion of data")
    prs.add_argument("--marker", nargs="+", required=False, choices=['.',',','*','+','o'], help="Maker per file")
    prs.add_argument("--output", default="tsne", help="Output name")
    prs.add_argument("--max-objective", action='store_true', help="Specify when objective should be MAXIMIZED instead of MINIMIZED (latter is default)")
    prs.add_argument("--rank-color", action='store_true', help="Darken color of points that are higher ranked, lighten color of points that are lower ranked")
    prs.add_argument("--seed", type=int, default=1234, help="Set seed for TSNE")
    prs.add_argument("--stepsize", type=float, default=0.1, help="Stepsize for mesh")
    prs.add_argument("--neighbors", type=int, default=1, help="Number of nearest neighbors")
    prs.add_argument("--video", action='store_true', help='Save as video switching ranks each frame')
    prs.add_argument("--fig-dims", metavar=("Xinches", "Yinches"), nargs=2, type=float,
                     default=plt.rcParams["figure.figsize"], help="Figure size in inches "
                     f"(default is {plt.rcParams['figure.figsize']})")
    prs.add_argument("--fig-pts", type=float, default=None, help="Specify figure size using LaTeX points and Golden Ratio")
    prs.add_argument("--format", choices=["png", "pdf", "svg", "jpeg"], default="pdf", help="Format to save outputs in")
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
    if len(args.quantile) == 1:
        args.quantile = [args.quantile[0] for _ in range(len(args.convert))]
    if len(args.quantile) != len(args.convert):
        raise ValueError("Require 1 global quantile or 1 quantile per converted file\n"+\
                         f"Quantiles: {args.quantile}"+"\n"+f"Files: {args.convert}")
    if args.marker is None:
        args.marker = []
    if args.marker == []:
        args.marker.append('.')
    while len(args.marker) < len(args.convert):
        args.marker.append(args.marker[-1])
    if args.fig_pts is not None:
        args.fig_dims = set_size(args.fig_pts)
    return args

def get_size(name, frame):
    if 'input' in frame.columns:
        return frame['input'].iloc[0]
    # Most definite to least certain order
    name = os.path.basename(name)
    for fname_size in ['_sm_', '_ml_', '_xl_', '_s_', '_m_', '_l_']:
        if fname_size in name.lower():
            return fname_size[1:-1]
    for fname_size in ['_sm','_ml','_xl','_s','_m','_l']:
        if fname_size in name.lower():
            return fname_size[1:]
    raise ValueError(f"Unsure of size for {name}")

def load(args):
    loaded = []
    for name in args.convert:
        frame = pd.read_csv(name)
        size = get_size(name, frame)
        frame = frame.sort_values(by='objective', ascending=not args.max_objective)
        #frame = frame.iloc[:min(len(frame),max(1,int(quant*len(frame))))]
        # Get params and add ranked objectives
        param_cols = sorted(set(frame.columns).difference({'objective','predicted','elapsed_sec'}))
        p_values = frame[param_cols]
        p_values.insert(len(p_values.columns), "rank", [_ for _ in range(1, len(p_values)+1)])
        #p_values.insert(len(p_values.columns), "rank", [_ for _ in range(len(p_values),0,-1)])
        p_values.insert(len(p_values.columns), "size", [size for _ in range(len(p_values))])
        loaded.append(p_values)
        print(f"Load {name} ==> {len(p_values)} rows")
    return loaded

def tsne_reduce(loaded, args):
    iloc_idxer = [len(_) for _ in loaded]
    stacked = pd.concat(loaded).drop(columns=['rank','size'])
    problem, attr = args.problem.rsplit('.',1)
    module = importlib.import_module(problem)
    space = module.__getattr__(attr).input_space
    skopt_space = skopt.space.Space(space)
    x_params = skopt_space.transform(stacked.astype('str')[sorted(stacked.columns)].to_numpy())
    tsne = TSNE(n_components=2, random_state=args.seed)
    print(f"TSNE reduces {x_params.shape} --> {(x_params.shape[0],2)}")
    new_values = tsne.fit_transform(x_params)
    new_loaded = []
    prev_idx = 0
    for idx, idx_end in enumerate(iloc_idxer):
        new_idx = prev_idx + idx_end
        new_frame = {
                     'x': new_values[prev_idx:new_idx, 0],
                     'y': new_values[prev_idx:new_idx, 1],
                     'z': loaded[idx]['rank'],
                     'label': loaded[idx]['size'],
                    }
        prev_idx = new_idx
        new_loaded.append(pd.DataFrame(new_frame))
        print(f"TSNE of {loaded[idx]['size'].iloc[0]} queued for plot")
    return new_loaded

def scale_relabel(label):
    known = {'S': 'Small',
             'SM': 'Small-Medium',
             'M': 'Medium',
             'ML': 'Medium-Large',
             'L': 'Large',
             'XL': 'Extra-Large',
             }
    if label.upper() in known.keys():
        return known[label.upper()]
    return label

def plot(loaded, args):
    fig, ax = plt.subplots(figsize=tuple(args.fig_dims))
    fig.set_tight_layout(True)
    color_maps = ['Yellows','Oranges','Reds','Blues','Greens','Purples','Greys', 'YlOrBr', 'PuRd', 'BuPu', 'YlOrRd', 'GnBu', 'OrRd', 'YlGnBu', 'YlGn',]
    knncolor = ['moccasin','coral','firebrick','lightcoral','powderblue','palegreen','plum','gainsboro']
    leg_handles = []
    marker_sizes = {'o': 20,
                    '*': 40,
                    '+': 40,
                    ',': 20,
                    '.': 20,
                }
    # KNN background?
    custom_cmap = matplotlib.colors.ListedColormap([knncolor[_] for _ in range(len(loaded))])
    combined = pd.concat(loaded)
    boundaries = [(combined[_].min()-1, combined[_].max()+1, args.stepsize) for _ in ['x','y']]
    classifier = neighbors.KNeighborsClassifier(args.neighbors, weights='distance')
    # Prepare classifier input
    X = np.stack((combined['x'],combined['y']),axis=1)
    Y = []
    for idx, length in enumerate(map(len,loaded)):
        Y.extend([idx]*length)
    Y = np.asarray(Y)
    classifier.fit(X,Y)
    # Mesh
    xx,yy = np.meshgrid(np.arange(*boundaries[0]), np.arange(*boundaries[1]))
    print("start classify")
    Z = classifier.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    print("stop classify")
    ax.pcolormesh(xx,yy,Z, cmap=custom_cmap)

    # Scatter points
    highlighted = []
    for (idx, line), cmap, quant in zip(enumerate(loaded), color_maps, args.quantile):
        if idx < len(args.marker):
            marker = args.marker[idx]
        else:
            marker='o'
        markersize = marker_sizes[marker]
        optimal_marker='x' if marker != 'x' else '*'
        if args.rank_color and len(line) > 1:
            quant_break = int(len(line['x']) * quant)
            # Indicated Optimal
            ax.scatter(line['x'].iloc[1:quant_break], line['y'].iloc[1:quant_break], color='black', label=line['label'].iloc[0], marker=marker, s=markersize)
            # Indicated Sub-optimal
            ax.scatter(line['x'].iloc[quant_break:], line['y'].iloc[quant_break:], color='white', label=line['label'].iloc[0], marker=marker, s=markersize)
            #ax.scatter(line['x'].iloc[1:], line['y'].iloc[1:], c=line['z'].iloc[len(line['z'])-2::-1], cmap=cmap, label=line['label'].iloc[0], marker=marker, s=markersize)
            # Line that goes to optimal marker
            x_line = ax.plot([0,line['x'].iloc[0]], [0,line['y'].iloc[0]], color='black', linewidth=0.5)
            # Optimal marker given a border even though markers don't have borders
            x_marker = ax.scatter(line['x'].iloc[0], line['y'].iloc[0], color='black', marker=optimal_marker, s=markersize*5, linewidth=5)
            x_marker = ax.scatter(line['x'].iloc[0], line['y'].iloc[0], color=cmap.rstrip('s').lower(), label='BEST'+line['label'].iloc[0], marker=optimal_marker, s=markersize*4, linewidth=3)
        else:
            x_marker = ax.scatter(line['x'].iloc[0], line['y'].iloc[0], color=cmap.rstrip('s').lower(), label='BEST'+line['label'].iloc[0], marker=optimal_marker, s=markersize*4, linewidth=3)
            ax.scatter(line['x'].iloc[1:], line['y'].iloc[1:], color=cmap.rstrip('s').lower(), label=line['label'].iloc[0], marker=marker, s=markersize)
            x_line = ax.plot([0,line['x'].iloc[0]], [0,line['y'].iloc[0]], color='black', linewidth=0.5)
        highlighted.append((x_marker,x_line[0]))
        leg_handles.append(matplotlib.lines.Line2D([0],[0],
                            marker=marker,
                            color='w',
                            label=scale_relabel(line['label'].iloc[0].upper()),
                            markerfacecolor=cmap.rstrip('s').lower(),
                            markersize=markersize))
    # Add origin lines
    ax.axhline(y=0, color='black', linewidth=0.1)
    ax.axvline(x=0, color='black', linewidth=0.1)
    # Labels, legends, save
    #ax.set_xlabel("TSNE dimension 1")
    #ax.set_ylabel("TSNE dimension 2")
    ax.legend(handles=leg_handles, loc="best", title='Scale')
    ax.set_xlim([min([min(line['x']) for line in loaded]), max([max(line['x']) for line in loaded])])
    ax.set_ylim([min([min(line['y']) for line in loaded]), max([max(line['y']) for line in loaded])])
    if not args.video:
        plt.savefig(f'{args.output}.{args.format}', format=args.format, bbox_inches='tight')
    else:
        max_rank = min([len(line['x']) for line in loaded])-1
        class animator:
            def __init__(self, highlight,loaded,max_rank):
                self.target_rank = 0
                self.max_rank = max_rank
                self.highlighted = highlight
                self.loaded = loaded
            def __call__(self, frame):
                if self.target_rank >= max_rank:
                    self.target_rank = 0
                else:
                    self.target_rank += 1
                rank = self.target_rank
                for hl, line in zip(self.highlighted, self.loaded):
                    try:
                        hl[0].set_offsets((line['x'].iloc[rank], line['y'].iloc[rank]))
                    except:
                        import pdb
                        pdb.set_trace()
                    hl[1].set_data(([0,line['x'].iloc[rank]], [0,line['y'].iloc[rank]]))
        make_animation = animator(highlighted,loaded,max_rank)
        anim = animation.FuncAnimation(fig,make_animation,frames=max_rank,interval=25)
        video = anim.to_html5_video()
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(f'{args.output}.mp4',writer=FFwriter)

if __name__ == "__main__":
    args = parse(build())
    loaded = load(args)
    loaded = tsne_reduce(loaded, args)
    plot(loaded, args)

