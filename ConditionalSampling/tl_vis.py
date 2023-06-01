import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np, pandas as pd
import operator

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--x-length', type=int, default=100, help="Number of x-points")
    prs.add_argument('--n-lines', type=int, default=3, help="Number of lines to plot")
    prs.add_argument('--func-order', type=int, default=2, help="Order of functions")
    prs.add_argument('--rounding', type=int, default=2, help="Rounding precision")
    prs.add_argument('--bias-var', type=float, default=30, help="Bias variation of funcs")
    prs.add_argument('--coeff-var', type=float, default=4, help="Coeff variation of funcs")
    prs.add_argument('--seed', type=int, default=1233, help="RNG seed")
    prs.add_argument('--dpi', type=int, default=300, help="DPI for figure")
    prs.add_argument('--manual-func', action='store_true', help="Let me write the func for you")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    np.random.seed(args.seed)
    return args

def gen_data(args):
    xs = np.resize(np.arange(args.x_length), (args.n_lines, args.x_length)).astype(float)
    ys = np.ones_like(xs).astype(float)
    if args.n_lines == 1:
        names = ['data']
    elif args.n_lines == 2:
        names = ['source', 'target']
    elif args.n_lines == 3:
        names = ['small', 'medium', 'large']
    else:
        names = [f'trend_{_}' for _ in range(1, 1+args.n_lines)]
    coeff, bias = None, None
    for axis_idx, tick_slice in enumerate(np.rollaxis(xs, 0)):
        if coeff is None:
            coeff = np.round(np.random.rand(args.func_order) / 2, args.rounding)
        else:
            # Nudge so that this data looks correlated and transferrable
            coeff_nudge = ((np.random.rand(1) - 0.5) / args.coeff_var) * np.ones_like(coeff)
            # Diminishing change for larger elements
            for idx in range(1, len(coeff)):
                coeff_nudge[idx] += (0.01 ** idx) * coeff_nudge[idx-1]
            # Coeff nudge should not change sign of coefficients
            #sign_adjust = np.where(np.sign(coeff) - np.sign(coeff_nudge) != 0)[0]
            #coeff_nudge[sign_adjust] *= -1/2
            coeff = np.round(coeff + coeff_nudge, args.rounding)
        if bias is None:
            bias = np.round((np.random.rand(args.func_order) - 0.5) * args.bias_var, args.rounding)
        else:
            bias_nudge = (np.random.rand(len(bias)) - 0.5) * args.bias_var
            bias = np.round(bias + bias_nudge, args.rounding)
        if args.func_order > 1:
            for idx in range(args.func_order):
                ys[axis_idx,:] *= (coeff[idx] * tick_slice) + bias[idx]
        else:
            ys[axis_idx,:] = coeff * tick_slice + bias
    # Usually makes sense -- order data based on minimum because these names convey more meaning
    if args.n_lines == 3:
        name_order = np.argsort([min(ys[axis_idx,:]) for axis_idx in range(args.n_lines)])
        ys = np.asarray([ys[axis_idx,:] for axis_idx in name_order])
    return xs, ys, names

def manual_func(args):
    args.x_length = 100
    args.n_lines = 3
    xs = np.reshape([-20,0,20,40,60,80,100,110] * 3, (3,8)).astype(float)
    ys = np.asarray([[5,10,15,10,8,12,14,14],
                     [5,15,25,15,10,14,15,15],
                     [5,20,40,20,10,25,26,26]]).astype(float)
    names = ['small', 'medium', 'large']
    return xs, ys, names

def main(args=None):
    if args is None:
        prs = build()
        args = parse(prs)
    fig, ax = plt.subplots()
    if args.manual_func:
        xs, ys, names = manual_func(args)
    else:
        xs, ys, names = gen_data(args)
    if args.manual_func:
        connect_coord = 2
        head_width = 2
        head_length = 1
        yy_connect = [14.47, 23.85, 37.66]
        yy_connect_counter = 0
    else:
        connect_coord = xs.shape[1]//2
        head_width = 2
        head_length = 100
    trends, dots = [], []
    indent = 20
    for xx, yy, name in zip(np.rollaxis(xs, 0), np.rollaxis(ys, 0), names):
        if args.manual_func:
            dots.append(ax.scatter(xx[connect_coord],yy_connect[yy_connect_counter], zorder=0))
            yy_connect_counter += 1
            verts = []
            for xx0, xx1, yy0, yy1 in zip(xx[:-1], xx[1:], yy[:-1], yy[1:]):
                temp_verts = [(xx0, yy0), (xx0+0.65*(xx1-xx0), yy0+0.65*(yy1-yy0)), (xx1,yy1)]
                verts.extend(temp_verts)
            verts = verts[1:-1]
            codes = [matplotlib.path.Path.MOVETO] + [matplotlib.path.Path.CURVE4] * (len(verts) - 1)
            path = matplotlib.path.Path(verts, codes)
            patch = matplotlib.patches.PathPatch(path, facecolor='none', edgecolor=dots[-1].get_facecolor()[0], zorder=0, label=name.capitalize())
            ax.add_patch(patch)
            trends.append(patch)
            #trends.extend(ax.plot(xx[1:-1],yy[1:-1],label=name.capitalize(), color=dots[-1].get_facecolor()[0], zorder=0))
        else:
            dots.append(ax.scatter(xx[connect_coord],yy[connect_coord], zorder=0))
            trends.extend(ax.plot(xx,yy,label=name.capitalize(), color=dots[-1].get_facecolor()[0], zorder=0))
    arrows = []
    for ax1 in range(args.n_lines-1,0,-1):
        ax0 = ax1 - 1
        if args.manual_func:
            arrows.append(ax.arrow(
                xs[ax0, connect_coord], yy_connect[ax0], # x0, y0
                0, yy_connect[ax1]-yy_connect[ax0], # dx, dy
                color = dots[ax0].get_facecolor()[0],
                head_width = head_width,
                head_length = head_length,
                length_includes_head = True,
                shape = 'full',
                linewidth = 1,
                zorder = 1,
                )
            )
        else:
            arrows.append(ax.arrow(
                xs[ax0, connect_coord], ys[ax0, connect_coord], # x0, y0
                0, ys[ax1, connect_coord] - ys[ax0, connect_coord], # dx, dy
                color = dots[ax0].get_facecolor()[0],
                head_width = head_width,
                head_length = head_length,
                length_includes_head = True,
                shape = 'full',
                linewidth = 1,
                zorder = 1,
                )
            )
    if args.manual_func:
        ax.legend([matplotlib.lines.Line2D([0],[0], color=dots[i].get_facecolor()[0]) for i in range(3)], [_.capitalize() for _ in names])
    else:
        ax.legend()
    ax.set_title("Arbitrary Transferrable $f(x)$")
    fig.set_dpi(args.dpi)
    plt.show()

if __name__ == '__main__':
    main()

