import argparse, sys

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--problems', nargs='+', type=str, required=True, help="Modules to look up")
    prs.add_argument('--size', type=str, required=False, default="L", help="Size class (usually can be omitted)")
    prs.add_argument('--attr', type=str, default='problem_class', help="Attribute to load")
    prs.add_argument('--show-problem', action='store_true', help="Prefix attribute with problem name loaded from (default off)")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.attr == 'problem_class':
        args.attr = args.size+'Problem'
    return args

def get_via_inspect(prob, attr):
    with open(prob, "r") as f:
        code = f.readlines()
    idx = min([i for i,line in enumerate(code) if line.lstrip().startswith("lookup_ival")])
    idx2 = idx+1+min([i for i,line in enumerate(code[idx:]) if line.rstrip().endswith("}")])
    data = code[idx:idx2]
    data[0] = data[0][data[0].index("=")+1:].lstrip()
    di = eval("".join(data))
    rdi = dict((v[0], k) for (k,v) in di.items())
    return rdi[attr.rstrip("Problem")]

def attr_transform(attrname, value):
    if attrname not in ['input_space_size']:
        return value
    elif attrname == 'input_space_size':
        prod = 1
        for (key, param) in value:
            if 'choices' in param:
                prod *= len(param['choices'])
            elif 'sequence' in param:
                prod *= len(param['sequence'])
            elif 'lower' in param and 'upper' in param:
                prod *= param['upper'] - param['lower']
            else:
                # Could warn here, but it'll generate way too much output
                # This catches when we don't know how to get a # of configurations
                # As Normal range is not necessarily defined with strict ranges and floats are floats
                continue
        return prod

def main(args):
    for prob in args.problems:
        if args.show_problem:
            print(f"{prob}: ", end='')
        # Attempt to import
        if prob.endswith('.py'):
            attr = args.attr
        else:
            prob, attr = prob.rsplit('.',1)
            prob += '.py'
        if attr == 'input_space_size':
            attr = 'input_space' # Have to fix later as the object doesn't instantiate to count
        try:
            # This way is a LOT faster for sizes, but requires source code to match particular patterns
            # that may not be implemented
            attr = attr_transform(args.attr, get_via_inspect(prob, attr))
            print(attr)
        except (KeyError,IndexError,AttributeError):
            # Direct import and load to get the attribute can potentially be a lot slower,
            # but it should always work as long as the object follows BaseProblem derivations
            from ytopt.search import util
            attr = attr_transform(args.attr, util.load_from_file(prob, attr))
            prob = prob.rstrip('.py')[prob.rindex('/')+1:]
            del sys.modules[prob]
            print(attr)

if __name__ == '__main__':
    main(parse(build()))

