import matplotlib
font = {'size': 14,
        'family': 'serif',
        }
lines = {'linewidth': 3,
         'markersize': 6,
        }
matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt, os, itertools
rcparams = {'axes.labelsize': 14,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            }
plt.rcParams.update(rcparams)

LIMIT_Y_TICKS = 10

def name_shortener(name):
    name = os.path.basename(name)
    if len(name.rsplit('.')) > 0:
        name = name.rsplit('.',1)[0]
    return name

drop_cols = ['predicted', 'elapsed_sec', 't0', 't1', 'isize']
ok_opacities = ['green','yellow','orange','blue','black','red']

def plotter_experiment(fig, ax, args):
    exhaust = pd.DataFrame({'objective': pd.read_csv(args.exhaust)['objective']}).sort_values(by='objective').reset_index(drop=True)
    ax = exhaust.plot(ax=ax, title='TBD', legend=False)
    for cand in args.candidate:
        candidate = pd.DataFrame({'objective': pd.read_csv(cand)['objective']}).sort_values(by='objective')
        if args.topk is not None:
            candidate = candidate.reset_index(drop=True).iloc[:args.topk]
        ax.scatter([int(len(exhaust)*np.mean(exhaust.to_numpy() <= _)) for _ in candidate.values],
                   [_ for _ in candidate.values],
                   label=f"{name_shortener(cand)}")
    return fig, ax

def plotter_lookup(fig, ax, args):
    exhaust = pd.read_csv(args.exhaust[0]).drop(columns=drop_cols, errors='ignore').sort_values(by='objective').reset_index(drop=True)
    ax = exhaust['objective'].plot(ax=ax, title='TBD', legend=False)
    for cand in args.candidate:
        candidate = pd.read_csv(cand).drop(columns=drop_cols, errors='ignore').sort_values(by='objective', ascending=False)
        if args.topk is not None:
            candidate = candidate.reset_index(drop=True).iloc[:args.topk]
        cand_cols = tuple([_ for _ in candidate.columns if _ != 'objective'])
        x,y,z = [], [], []
        permit_win_at = len(candidate)//2 #if 'gptune' in cand else 1
        random_objectives = list(candidate.sort_index()['objective'].iloc[:permit_win_at])
        gptune_objectives = {random_objectives.index(min(random_objectives)): min(random_objectives)}
        for cand_row in candidate.iterrows():
            # Find if this was a W for GPTune over its random sampling period or not
            win = cand_row[0] >= permit_win_at
            # STRICT: Must improve over best random or best known so far
            #win = win and cand_row[1]['objective'] < gptune_objectives[max([k for k in gptune_objectives.keys() if k < cand_row[0]])]
            # LAX: Must improve over best random
            win = win and cand_row[1]['objective'] < list(gptune_objectives.values())[0]
            #if win:
            #    gptune_objectives[cand_row[0]] = cand_row[1]['objective']
            z.append(int(win))
            # Get the specific columns we want
            cand_row = cand_row[1][list(cand_cols)]
            search_equals = tuple(cand_row.values)
            n_matching_columns = (exhaust[list(cand_cols)] == search_equals).sum(1)
            full_match_idx = np.where(n_matching_columns == len(cand_cols))[0]
            match_data = exhaust.iloc[full_match_idx]
            x.append(match_data.index[0])
            y.append(match_data['objective'][x[-1]])
        print(cand, len(x), sorted(x))
        # Search for equals, and plot that from exhaust instead
        ax.scatter(x,y,label=f"{name_shortener(cand)}")
        print(f"GPTUNE improvements after sampling: {sum(z)}")
    return fig, ax

def plotter_multi_mean_median(fig, ax, args):
    exhausts = [pd.read_csv(_).drop(columns=drop_cols, errors='ignore').sort_values(by='objective').reset_index(drop=True) for _ in args.exhaust]
    # Drop 1.0's -- bad evaluations
    exhausts = [_.drop(_[_['objective']==1.0].index).reset_index(drop=True) for _ in exhausts]
    colors = [_['color'] for _ in plt.rcParams['axes.prop_cycle']]
    names = [_.rsplit('_',1)[1].split('.',1)[0] for _ in args.exhaust]
    ax = exhausts[0]['objective'].plot(ax=ax,legend=False, color=colors[0], label="Observed Performance")#f"{names[0]} Objective")
    mean, median = exhausts[0]['objective'].mean(), exhausts[0]['objective'].median()
    #mean_line = ax.plot([_ for _ in range(len(exhausts[0]))], [mean for _ in range(len(exhausts[0]))], label=f'{names[0]} Mean', linestyle='--', color=colors[1])
    #median_line = ax.plot([_ for _ in range(len(exhausts[0]))], [median for _ in range(len(exhausts[0]))], label=f'{names[0]} Median', linestyle='--', color=ax.lines[0].get_color())#colors[2])
    median_line = ax.plot([_ for _ in range(len(exhausts[0]))], [median for _ in range(len(exhausts[0]))], label=f'Median Performance', linestyle='--', color=ax.lines[0].get_color())#colors[2])
    #nearest_mean = np.argmin(abs(exhausts[0]['objective']-mean))
    nearest_median = np.argmin(abs(exhausts[0]['objective']-median))
    #ax.scatter(x=[nearest_mean, nearest_median], y=[mean, median],
    #           c=[mean_line[0].get_color(), median_line[0].get_color()], s=[32,32])
    ax.scatter(x=[nearest_median], y=[median],
               c=[median_line[0].get_color()], s=[32])
    one_percent = int(0.01*len(exhausts[0]))
    #ax.fill_between([_ for _ in range(one_percent)], [0 for _ in range(one_percent)], exhausts[0]['objective'].iloc[:one_percent],
    #                alpha=0.2, color=ax.lines[0].get_color())
    #print(f"Mean: {mean} closest to rank {nearest_mean}")
    print(f"Median: {median} closest to rank {nearest_median}")
    color_idx = 3
    handles, labels = ax.get_legend_handles_labels()
    for exhaust, name in zip(exhausts[1:], names[1:]):
        bonus_ax = ax.twinx()
        bonus_ax.plot(exhaust['objective'],color=colors[color_idx], label=f"{name} Objective")
        color_idx += 1
        mean, median = exhaust['objective'].mean(), exhaust['objective'].median()
        #mean_line = bonus_ax.plot([_ for _ in range(len(exhaust))], [mean for _ in range(len(exhaust))], label=f'{name} Mean', linestyle='--', color=colors[color_idx])
        color_idx += 1
        median_line = bonus_ax.plot([_ for _ in range(len(exhaust))], [median for _ in range(len(exhaust))], label=f'{name} Median', linestyle='--', color=bonus_ax.lines[0].get_color())#colors[color_idx])
        #nearest_mean = np.argmin(abs(exhaust['objective']-mean))
        nearest_median = np.argmin(abs(exhaust['objective']-median))
        #bonus_ax.scatter(x=[nearest_mean, nearest_median], y=[mean, median],
        #           c=[mean_line[0].get_color(), median_line[0].get_color()], s=[32,32])
        bonus_ax.scatter(x=[nearest_median], y=[median],
                   c=[median_line[0].get_color()], s=[32])
        one_percent = int(0.01*len(exhaust))
        bonus_ax.fill_between([_ for _ in range(one_percent)], [0 for _ in range(one_percent)], exhaust['objective'].iloc[:one_percent],
                              alpha=0.2, color=bonus_ax.lines[0].get_color())
        #print(f"Mean: {mean} closest to rank {nearest_mean}")
        print(f"Median: {median} closest to rank {nearest_median}")
        bonus_ax.set_yscale('log')
        bonus_ax.set_ylabel(f"{name} Objective Time (seconds)")
        bonus_handles, bonus_labels = bonus_ax.get_legend_handles_labels()
        handles.extend(bonus_handles)
        labels.extend(bonus_labels)
    ax.set_xlabel("Configuration")
    #ax.set_xlabel("Performance Rank of Configuration (Lower is Better)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_ylabel(f"{names[0]} Objective Time (seconds)")
    ax.set_ylabel(f"Performance Objective Time (seconds)")
    ax.legend(handles, labels)
    args.no_legend = True
    return fig, ax

def plotter_mean_median(fig, ax, args):
    exhaust = pd.read_csv(args.exhaust).drop(columns=drop_cols, errors='ignore').sort_values(by='objective').reset_index(drop=True)
    # Drop 1.0's -- bad evaluations
    exhaust = exhaust.drop(exhaust[exhaust['objective']==1.0].index).reset_index(drop=True)
    ax = exhaust['objective'].plot(ax=ax, title='TBD', legend=False)
    mean, median = exhaust['objective'].mean(), exhaust['objective'].median()
    mean_line = ax.plot([_ for _ in range(len(exhaust))], [mean for _ in range(len(exhaust))], label='mean', linestyle='--')
    median_line = ax.plot([_ for _ in range(len(exhaust))], [median for _ in range(len(exhaust))], label='median', linestyle='--')
    nearest_mean = np.argmin(abs(exhaust['objective']-mean))
    nearest_median = np.argmin(abs(exhaust['objective']-median))
    ax.scatter(x=[nearest_mean, nearest_median], y=[mean, median],
               c=[mean_line[0].get_color(), median_line[0].get_color()], s=[32,32])
    ax.set_xlabel("Performance Rank of Configuration (Lower is Better)")
    ax.set_ylabel("Objective Time (seconds)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    print(f"Mean: {mean} closest to rank {nearest_mean}")
    print(f"Median: {median} closest to rank {nearest_median}")
    return fig, ax

def plotter_heat_map(fig, ax, args):
    global drop_cols
    exhaust = pd.read_csv(args.exhaust).drop(columns=drop_cols, errors='ignore').sort_values(by='objective').reset_index(drop=True)
    N_EXHAUST = len(exhaust)
    # Supplementary data determines RELEVANCE SCORES
    supplementary = [pd.read_csv(supp).sort_values(by='objective') for supp in args.supplementary]
    supplementary = pd.concat([_.iloc[:int(args.topsupp*len(_))] for _ in supplementary])

    # All allowable parameters are detailed in exhaustive data
    COLUMNS = [_ for _ in exhaust.columns if _ != 'objective']
    N_COLUMNS = len(COLUMNS)
    parameter_sets = [list(set(exhaust[_])) for _ in COLUMNS]
    longest_set = max(map(len,parameter_sets))

    # Determine occurrence counts over the supplementary data
    counts_from_supplementary = [[supplementary[col].tolist().count(val) for val in parameter_sets[idx]]
                                 for idx, col in enumerate(COLUMNS)]
    npy_counts_from_supplementary = -1 * np.ones((N_COLUMNS, longest_set))
    for idx, count in enumerate(counts_from_supplementary):
        npy_counts_from_supplementary[idx,:len(count)] = count
    # Maximum count that could appear for a value
    max_supplementary_count = len(supplementary) * N_COLUMNS

    # Give a relevance score to each exhaustive configuration
    relevance = np.zeros(N_EXHAUST)
    for idx, config in exhaust.iterrows():
        # Use list to figure out what parameter value is set for each column
        relevance_indices = [li.index(val) for li, val in zip(parameter_sets, config)]
        relevance[idx] = sum([npy_counts_from_supplementary[i,j] for (i,j) in enumerate(relevance_indices)])
        # Normalize
        relevance[idx] /= max_supplementary_count
    # Secondary normalization -- maybe only need one of them??
    relevance = (relevance-min(relevance))/(max(relevance)-min(relevance))
    # Descending order of relevance
    relevance_index = np.argsort(-relevance)

    # Make sure ALL values get bucketed
    for required in [0.0, 1.0]:
        if required not in args.buckets:
            args.buckets.append(required)
    # Splice indices for buckets -- may be uneven so cannot be ndarray
    args.buckets = sorted(args.buckets)
    # Slices are based on proportional length
    bucket_slices = [(int(start*N_EXHAUST), int(stop*N_EXHAUST)) for (start, stop) in zip(args.buckets[:-1],args.buckets[1:])]
    # Slices convert to quantiles
    buckets = [relevance_index[bucket_start:bucket_stop] for (bucket_start, bucket_stop) in bucket_slices]

    # Density measure on buckets
    density = np.zeros((len(args.buckets), N_EXHAUST))
    width = N_EXHAUST // 2
    # Weights towards density are based on proximity -- symmetric
    denominator = np.asarray([1/_ for _ in range(width,0,-1)]+[1]+[1/_ for _ in range(1,width+1)])
    denom_density = denominator.sum()
    for idx, bucket in enumerate(args.buckets[:-1]):
        import time
        start = time.time()
        # Presence represented as binary indicator
        indicator = np.zeros(N_EXHAUST)
        try:
            indicator[buckets[idx]] = 1
        except:
            import pdb
            pdb.set_trace()
        # Iteration boundaries
        left = -1
        right = width
        # SPLIT INTO 3 CASE LOOPS -- WRITING ONE LOOP IS REALLY HARD TO BE SEMANTICALLY CORRECT -- NO TANGIBLE PERFORMANCE GAIN TO BE MADE
        for it in range(width):
            left += 1
            # Mask of values is LEFT PADDED to match denominator
            mask = np.hstack((np.zeros(width-left), indicator[ : it+1+right]))
            density[idx,it] = (mask * denominator).sum() / denom_density
        if N_EXHAUST % 2 == 0:
            # Middle of an even-length series (LIKELY) has special handling since width on both sides INCLUDES the element itself
            it += 1
            right -= 1
            mask = np.hstack((indicator, np.zeros(1)))
            density[idx,it] = (mask * denominator).sum() / denom_density
        # Right loop picks up where we left off -- state may be adjusted above in case of even-length
        for it in np.arange(it+1, N_EXHAUST):
            right -= 1
            mask = np.hstack((indicator[it-width : ], np.zeros(width-right)))
            density[idx,it] = (mask * denominator).sum() / denom_density
        # Normalize each bucket's density
        density[idx] = (density[idx]-min(density[idx]))/(max(density[idx])-min(density[idx]))
        stop = time.time()
        print(f"Density calculation took {stop-start}")

    # INTENDED REPRESENTATION
    # X : Configurations
    x = np.arange(N_EXHAUST)
    # Y : Buckets
    y = args.buckets[::-1]
    # HEAT : Density Measure per Bucket (reverse iterate first axis to align with plot ascending direction)
    vertical_scale = density.shape[1] // density.shape[0]
    heat = np.repeat(density[::-1,:], vertical_scale, axis=0)

    ax.set_xticks([_ for _ in x if _ % 1000 == 0], labels=[str(_//1000)+'K' for _ in x if _ % 1000 == 0])
    ax.set_xlabel("Performance Rank of Configuration (Lower is Better)")
    if args.collapse_heat:
        collapsed = np.matmul(args.buckets, density) / sum(args.buckets)
        ax.plot(x, collapsed, label=f"Weighted probability based on buckets {args.buckets}")
    else:
        # Add 'auto' aspect so that the disparity between |X| and |Y| do not break the plot (tall pixels incoming)
        im = ax.imshow(heat, aspect='equal')
        if len(y) < LIMIT_Y_TICKS:
            ax.set_yticks(vertical_scale * np.arange(len(y)) + (vertical_scale/2), labels=y)
            if not args.no_minor_lines:
                ax.set_yticks(vertical_scale * np.arange(len(y)), minor=True)
        else:
            fair_mod = len(y) // LIMIT_Y_TICKS
            shorter_y = np.asarray([_ for idx,_ in enumerate(y) if idx % fair_mod == 0])
            # Rescaling required since we're omitting points
            vertical_scale *= len(y) / len(shorter_y)
            ax.set_yticks(vertical_scale * np.arange(len(shorter_y)) + (vertical_scale/2), labels=shorter_y)
            if not args.no_minor_lines:
                ax.set_yticks(vertical_scale * shorter_y, minor=True)
        ax.set_ylabel("Relevance Quantile (Lower is More Likely)")
        # Add the density color bar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Density of Preference (Higher is More Preferred)", rotation=-90, va='bottom')
        # Create the white grid
        ax.grid(which='minor',color='w',linestyle='-',linewidth=2, axis='y')

    return fig, ax

def plotter_implied_area(fig,ax,args):
    global drop_cols
    exhaust = pd.read_csv(args.exhaust).drop(columns=drop_cols, errors='ignore').sort_values(by='objective').reset_index(drop=True)
    supplementary = pd.concat([pd.read_csv(supp).sort_values(by='objective').iloc[:int(args.topsupp*len(pd.read_csv(supp)))] for supp in args.supplementary])
    ax = exhaust['objective'].plot(ax=ax, title=f'Area implied by {", ".join([name_shortener(c) for c in args.candidate])}', legend=False)
    for cand_id, cand in enumerate(args.candidate):
        candidate = pd.read_csv(cand).drop(columns=drop_cols, errors='ignore').sort_values(by='objective', ascending=False)
        if args.topk is not None:
            candidate = candidate.reset_index(drop=True).iloc[:args.topk]
        cand_cols = tuple([_ for _ in candidate.columns if _ != 'objective'])
        if args.problem is None:
            allowed = [set(supplementary[_]) for _ in cand_cols]
            print("WARNING: Allowable configurations based upon supplementary data rather than problem permissible values")
        else:
            ispace = args.problem.input_space
            def seqchoice(v):
                if hasattr(v, 'sequence') and v.sequence is not None:
                    return v.sequence
                elif hasattr(v, 'choices') and v.choices is not None:
                    return v.choices
                else:
                    raise ValueError(f"No non-None sequence or choice attribute for {v}")
            allowed = [set(seqchoice(ispace[p])) for p in ispace.get_hyperparameter_names()]
            print("Allowable configurations based upon {args.problem}")
        supplementary_scores = [[list(supplementary[list(cand_cols)[col]].astype(str)).count(str(val)) for val in allowed[col]] for col in range(len(allowed))]
        x,y,z = [], [], []
        possible_specs = [_ for _ in itertools.product(*allowed)]
        save_name = f"exhaust_cache_for_{args.problem.name}_with_{'&&'.join([''.join(_.split('.')[:-1]).replace('/','#') for _ in args.supplementary])}@{args.topsupp}"
        if os.path.exists(save_name+'.npz'):
            loaded = np.load(save_name+'.npz')
            print(f"Loaded cache based on arguments: {save_name}.npz")
            x,y,z = [loaded[_] for _ in loaded.files]
        else:
            for spec_idx, spec in enumerate(possible_specs):
                if spec_idx % 10 == 0:
                    print(f" Compute global relevance: {100*spec_idx/len(possible_specs):.2f}% ", end='\r')
                search_equals = tuple([str(_) for _ in spec])
                n_matching_columns = (exhaust[list(cand_cols)].astype(str) == search_equals).sum(1)
                full_match_idx = np.where(n_matching_columns == len(cand_cols))[0]
                match_data = exhaust.iloc[full_match_idx]
                x.append(match_data.index[0])
                y.append(match_data['objective'][x[-1]])
                relevance_lookup_idx = [list([str(_) for _ in allowed[cidx]]).index(str(val)) for (cidx, val) in enumerate(search_equals)]
                #relevance_lookup_idx = [list(set(supplementary[col].astype(str))).index(str(val)) for (col,val) in zip(cand_cols, search_equals)]
                relevance = sum([supplementary_scores[col][idx] for (col,idx) in zip(range(len(allowed)),relevance_lookup_idx)])/(len(supplementary)*len(allowed))
                z.append(relevance) # Notion of how much the candidate liked this set of parameters
            print()
            np.savez(save_name, x,y,z)
        relevance = np.asarray(z)
        relevance = (relevance-min(relevance))/(max(relevance)-min(relevance))
        order = np.argsort(-np.asarray(x))
        if args.buckets is not None:
            # STATIC LIST
            global ok_opacities
            reverse_bucket = dict((k,v) for (k,v) in zip(ok_opacities, [1.0]+sorted(args.buckets)[::-1]))
            reverse_reverse_bucket = dict((v,k) for (k,v) in reverse_bucket.items())
            print(reverse_bucket)
            idx = set([_ for _ in range(len(relevance))])
            buckets = []
            bucket_names = sorted(args.buckets)
            for b in bucket_names:
                buckets.append([_ for _ in np.where(relevance <= np.quantile(relevance, q=b))[0] if _ in idx])
                idx = idx.difference(set().union(*buckets))
            if len(idx) > 0:
                buckets.append(list(idx))
                bucket_names.append(1.0)
            buckets.reverse() # Put in best to worse order
            bucket_names.reverse()
            output = []
            for idx in range(len(z)):
                bid = [idx in b for b in buckets].index(True)
                output.append(ok_opacities[bid])
            #c1 = np.asarray([251,87,93])/255
            #c2 = np.asarray([108,183,137])/255
            c2 = np.asarray([255,255,255])/255
            # Density based bucket lines
            #import pdb
            #pdb.set_trace()
            for bucket_id, bucket in enumerate(buckets):
                y_height = len(buckets)-bucket_id
                print(f"bucket {bucket_names[bucket_id]} adds y-height {y_height}")
                length = len(x)
                width = length // 2
                # Count a 1 if this x-value is in the bucket
                bucket_value = [1 if val in bucket else 0 for val in range(length)]
                density_measure = np.zeros(length)
                # Weights towards density are based on proximity -- symmetric
                denominator = np.asarray([1/_ for _ in range(width,0,-1)]+[1]+[1/_ for _ in range(1,width+1)])
                # Maximum possible density to measure
                denom_density = denominator.sum()
                # Iteration boundaries
                left = -1
                right = width
                import time
                start = time.time()
                # SPLIT INTO 3 CASE LOOPS -- WRITING ONE LOOP IS REALLY HARD TO BE SEMANTICALLY CORRECT -- NO TANGIBLE PERFORMANCE GAIN TO BE MADE
                for it in range(width):
                    left += 1
                    # Mask of values is LEFT PADDED to match denominator
                    mask = np.hstack((np.zeros(width-left), bucket_value[ : it+1+right]))
                    density_measure[it] = (mask * denominator).sum() / denom_density
                if length % 2 == 0:
                    # Middle of an even-length series (LIKELY) has special handling since width on both sides INCLUDES the element itself
                    it += 1
                    right -= 1
                    mask = np.hstack((bucket_value, np.zeros(1)))
                    density_measure[it] = (mask * denominator).sum() / denom_density
                # Right loop picks up where we left off -- state may be adjusted above in case of even-length
                for it in np.arange(it+1, length):
                    right -= 1
                    mask = np.hstack((bucket_value[it-width : ], np.zeros(width-right)))
                    density_measure[it] = (mask * denominator).sum() / denom_density
                # Normalize densities so color differences display more fully
                stop = time.time()
                print(f"Density calculation took {stop-start}")
                density_measure = (density_measure-min(density_measure))/(max(density_measure)-min(density_measure))
                # Need to define c1 and c2 for mix-ins per bucket somehow as 0-1 based rgb ndarrays
                vertices = np.zeros((len(x),2,2))
                colors = np.zeros((len(x),3))
                # Need to define the y-height per bucket somehow to represent things (follow the curvature? flat height?)
                for i in range(length):
                    vertices[i] = [[i,exhaust.iloc[i]['objective']+y_height],[i+1,exhaust.iloc[min(length-1,i+1)]['objective']+y_height]]
                    c1 = np.asarray(matplotlib.colors.to_rgb(reverse_reverse_bucket[bucket_names[bucket_id]]))
                    colors[i] = ((1-density_measure[i])*c1)+((density_measure[i])*c2)
                lc = matplotlib.collections.LineCollection(vertices, colors=colors, linewidth=4)
                ax.add_collection(lc)
    return fig, ax

def add_default_line(ax, args):
    try:
        from problem import S
    except ImportError:
        print("Unable to load problem.py as module -- no default line generated")
    exhaust = pd.read_csv(args.exhaust).drop(columns=drop_cols, errors='ignore').sort_values(by='objective').reset_index(drop=True)
    cand_cols = tuple([_ for _ in exhaust.columns if _ != 'objective'])
    default_config = S.input_space.get_default_configuration().get_dictionary()
    # If only pandas had a simple way to map dictionaries to an existing dataframe's types....
    search_equals = tuple(pd.DataFrame(default_config, index=[0]).astype(exhaust.dtypes[list(cand_cols)]).iloc[0].values)
    n_matching_columns = (exhaust[list(cand_cols)] == search_equals).sum(1)
    full_match_idx = np.where(n_matching_columns == len(cand_cols))[0]
    match_data = exhaust.iloc[full_match_idx]
    if plotter_mean_median in args.func:
        print(f"Default: {match_data['objective'].iloc[0]} at rank {match_data.index[0]}")
    ax.plot(ax.lines[0].get_xdata(), [match_data['objective'].iloc[0] for _ in ax.lines[0].get_xdata()], label='Default -O3')

ncalls = 0
def common(func, args):
    fig, ax = func(*plt.subplots(figsize=tuple(args.fig_dims), dpi=args.dpi), args)
    fig.set_tight_layout(True)
    if args.default:
        add_default_line(ax, args)
    if not args.no_legend:
        ax.legend()
    global ncalls
    ncalls += 1
    if args.auto_fill:
        if args.xmin is not None and args.ymin is None:
            args.ymin = ax.lines[0]._y[min(np.where(ax.lines[0]._x <= args.xmin)[0])]
        if args.ymin is not None and args.xmin is None:
            args.xmin = ax.lines[0]._x[min(np.where(ax.lines[0]._y >= args.ymin)[0])]
        if args.xmax is not None and args.ymax is None:
            args.ymax = ax.lines[0]._y[max(np.where(ax.lines[0]._x <= args.xmax)[0])]
        if args.ymax is not None and args.xmax is None:
            args.xmax = ax.lines[0]._x[max(np.where(ax.lines[0]._y <= args.ymax)[0])]
    if args.xmax is not None or args.xmin is not None:
        ax.set_xlim([args.xmin, args.xmax])
    if args.ymax is not None or args.ymin is not None:
        ax.set_ylim([args.ymin, args.ymax])
    if args.title is not None:
        ax.set_title(args.title)
    print(f"Saving figure to {args.figname}_{ncalls}.{args.format}")
    plt.savefig(f"{args.figname}_{ncalls}.{args.format}", format=args.format, bbox_inches='tight')

def build():
    plotter_funcs = dict((k,v) for (k,v) in globals().items() if k.startswith('plotter_') and callable(v))
    prs = argparse.ArgumentParser()
    prs.add_argument('--exhaust', type=str, nargs="*", help="Exhaustive evaluation to compare against")
    prs.add_argument('--candidate', type=str, nargs="*", help="Candidate evaluation to compare to exhaustion")
    prs.add_argument('--supplementary', type=str, nargs="*", help="Supplementary data for relevance calculation")
    prs.add_argument('--topsupp', type=float, default=0.3, help="Top%% of supplementary data to use")
    prs.add_argument('--func', choices=[_[8:] for _ in plotter_funcs.keys()], nargs='+', required=True, help="Function to use")
    prs.add_argument('--figname', type=str, default="plot", help="Figure name")
    prs.add_argument("--fig-dims", metavar=("Xinches", "Yinches"), nargs=2, type=float,
                     default=plt.rcParams["figure.figsize"], help="Figure size in inches "
                     f"(default is {plt.rcParams['figure.figsize']})")
    prs.add_argument("--fig-pts", type=float, default=None, help="Specify figure size using LaTeX points and Golden Ratio")
    prs.add_argument("--format", choices=["png", "pdf", "svg","jpeg"], default="pdf", help="Format to save outputs in")
    prs.add_argument('--xmax', type=float, default=None, help="Set xlimit maximum")
    prs.add_argument('--xmin', type=float, default=None, help="Set xlimit minimum")
    prs.add_argument('--ymax', type=float, default=None, help="Set ylimit maximum")
    prs.add_argument('--ymin', type=float, default=None, help="Set ylimit minimum")
    prs.add_argument('--auto-fill', action='store_true', help="Infer better xlimit/ylimit from partial specification")
    prs.add_argument('--title', type=str, default=None, help="Provide a figure title")
    prs.add_argument('--no-legend', action='store_true', help="Omit legend")
    prs.add_argument('--topk', type=int, default=None, help="Only plot top k performing candidate points")
    prs.add_argument('--default', action='store_true', help="Attempt to infer a default configuration from problem.py")
    prs.add_argument('--buckets', type=float, nargs='*', default=None, help="# Buckets for functions that use them")
    prs.add_argument('--n-buckets', type=int, default=None, help="Ensure buckets list includes auto-range values for N entries")
    prs.add_argument('--no-minor-lines', action='store_true', help="Disable minor axis lines")
    prs.add_argument('--problem', type=str, default=None, help='Problem for importing information')
    prs.add_argument('--attribute', type=str, default=None, help='Attribute to fetch a problem instance for information')
    prs.add_argument('--collapse-heat', action='store_true', help="Make heat as a CDF rather than a 2D map")
    prs.add_argument('--dpi', type=float, default=matplotlib.rcParams['figure.dpi'], help="Figure DPI (default: %(default)s)")
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
    if len(args.exhaust) == 1:
        args.exhaust = args.exhaust[0]
    if args.n_buckets is not None:
        args.buckets = [_/args.n_buckets for _ in range(args.n_buckets)]
    if args.buckets is None:
        args.buckets = []
    # Some plots are limited by known colors
    if len(args.buckets) > len(ok_opacities) and 'implied_area' in args.func:
        raise ValueError(f"Due to color limitations, 'implied_area' can only support {len(ok_opacities)} buckets (given {len(args.buckets)})")
    plotter_funcs = dict((k,v) for (k,v) in globals().items() if k.startswith('plotter_') and callable(v))
    args.func = [plotter_funcs['plotter_'+func] for func in args.func]
    if args.problem is not None and args.attribute is not None:
        import importlib
        item = importlib.import_module(args.problem)
        for attribute in args.attribute.split('.'):
            item = getattr(item, attribute)
        args.problem = item
        del args.attribute
    elif args.problem is not None or args.attribute is not None:
        raise ValueError(f"Must specify BOTH problem and attribute")
    if type(args.exhaust) is str:
        args.exhaust = [args.exhaust]
    if args.fig_pts is not None:
        args.fig_dims = set_size(args.fig_pts)
    return args

def main(args=None):
    if args is None:
        args = parse(build())
    for func in args.func:
        common(func, args)

if __name__ == '__main__':
    main()

