import pandas as pd, numpy as np, argparse

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--files', nargs='+', type=str, help="Paths to load for analysis")
    prs.add_argument('--minmax', action='store_true', help="Instead of raw data, present as min/max per target size")
    prs.add_argument('--compress', action='store_true', help="Present compressed follow-up on min/max")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    return args

def main(args=None):
    if args is None:
        args = parse(build())
    sources = ['source_size', 'source_objective']
    targets = ['target_size', 'target_objective']
    display = ['source_size', 'target_size', 'target_objective', 'applied_speedup']
    compression = np.zeros((len(args.files),3,2))
    stackHistory = []
    for fid, file in enumerate(args.files):
        d = pd.read_csv(file)
        lookup = dict(d[sources].iloc[np.where(d[sources[0]].sub(d[targets[0]]).values==0)[0]].values)
        d['applied_speedup'] = d[targets].apply(lambda r: lookup[r[0]]/r[1], axis=1, result_type='expand')
        history = []
        print(file)
        tsizes = sorted(set(d['target_size']))
        for tid, tsize in enumerate(tsizes):
            relevant = d[display].iloc[np.where(d['target_size']==tsize)[0]]
            # Prepare for compression
            idxmin = np.argmin(relevant['applied_speedup'])
            idxmax = np.argmax(relevant['applied_speedup'])
            compression[fid,tid,0] = relevant.iloc[idxmin]['applied_speedup']
            compression[fid,tid,1] = relevant.iloc[idxmax]['applied_speedup']
            history.append(relevant[relevant['source_size'].isin(tsizes)]['applied_speedup'].values)
            if args.minmax:
                print(f"MIN: {pd.DataFrame([relevant.iloc[idxmin]])}")
                print(f"MAX: {pd.DataFrame([relevant.iloc[idxmax]])}")
            else:
                print(relevant)
        stackHistory.append(history)
    if args.compress:
        if args.minmax:
            print(compression.mean(axis=0, dtype=np.float64))
        else:
            print(np.asarray(stackHistory).mean(axis=0))

if __name__ == '__main__':
    main()

