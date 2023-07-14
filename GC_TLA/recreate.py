import argparse
import pandas as pd
import pathlib
import re

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--csv", nargs="+", required=True,
                     help="File(s) to read and replicate")
    prs.add_argument("--template", required=True,
                     help="Template file to use for replication")
    prs.add_argument("--output-dir", default=".",
                     help="Directory where recreated files are placed (default: %(default)s)")
    return prs

def parse(prs=None, args=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    if type(args.csv) is str:
        args.csv = [args.csv]
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args

def export(data, args):
    with open(args.template, 'r') as f:
        lines = f.readlines()
    template_path = pathlib.Path(args.template)
    template_ext = template_path.suffix[1:]
    template_base = template_path.stem
    triggers = set(data.columns).difference({'fname','objective','elapsed_sec'})
    triggers = [f"#{t.upper()}" for t in triggers]
    for idx, row in data.iterrows():
        writeout = f"{args.output_dir}/{template_base}_{row['fname']}_{idx}.{template_ext}"
        with open(writeout, 'w') as w:
            for line in lines:
                for idx, trigger in enumerate(triggers):
                    while re.search(trigger, line):
                        foundGroups = []
                        for m in re.finditer(trigger, line):
                            match = m.group(0)
                            if match in foundGroups:
                                continue
                            line = re.sub(match, str(row[trigger.lower()[1:]]), line)
                            foundGroups.append(match)
                w.write(line)

def main(args=None):
    args = parse(args=args)
    loaded = []
    for fname in args.csv:
        try:
            frame = pd.read_csv(fname)
            fname = pathlib.Path(fname).stem
            frame.insert(0, 'fname', [fname] * len(frame))
            loaded.append(frame)
        except:
            print(f"Failed to load csv: {fname}")
    loaded = pd.concat(loaded)
    export(loaded, args)

if __name__ == '__main__':
    main()

