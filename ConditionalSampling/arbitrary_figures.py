import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches, colors
import seaborn as sns
import numpy as np
import pandas as pd
import argparse

color_palette = colors.TABLEAU_COLORS
default_color = color_palette[list(color_palette.keys())[0]]
highlight_color = color_palette[list(color_palette.keys())[1]]
lighter_default_color = '#9cd5fd' # Manually lightened tab:blue
darker_default_color = '#174d72' # Manually darkened tab:blue
highlight_hex = "#ff7f0e" # Same as tab:orange

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--conditional', action='store_true', help="Draw the figure indicating conditional sampling (default: unconditional)")
    prs.add_argument('--show', action='store_true', help="Display figure instead of saving")
    prs.add_argument('--output_path', default="Assets/ConditionalExample.png", help="Figure save path (default: %(default)s)")
    return prs

def parse(prs=None, args=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    args.unconditional = not args.conditional
    return args

def anchor_points(augment=False):
    x0_xy = [78, 0] #75, 0.3
    if augment:
        x0_xy = [88.5, 0] # 0.2
    y_xy = [23, 0] #0.25
    if augment:
        y_xy = [16.25, 0] # 0.1
    size_xy = [1.9125, 0.5]
    covariance_xy = [0.5, 0.5]
    if augment:
        covariance_xy = [0.75, 0.75]
    return x0_xy, y_xy, size_xy, covariance_xy

def add_conditional_lines(fig, axis_x0, axis_y, axis_size, axis_covariance):
    xyX0, xyY, xySize, xyCovariance = anchor_points(augment=True)
    covariance_range = np.asarray([xyCovariance, xyCovariance])
    covariance_range[0] -= 0.25
    covariance_range[1] += 0.25
    #axis_x0.plot(*xyX0, "o", color=default_color)
    #axis_y.plot(*xyY, "o", color=default_color)
    #axis_size.plot(*xySize, "o", color=highlight_color)
    # More advanced coloring for covariance line
    axis_covariance.plot([0,covariance_range[0,0]],[0,covariance_range[0,1]], color=darker_default_color)
    axis_covariance.plot(covariance_range[:,0],covariance_range[:,1], color=highlight_color)
    axis_covariance.plot([covariance_range[1,0],1],[covariance_range[1,1],1], color=darker_default_color)
    axis_covariance.plot(*xyCovariance, "o", zorder=2, color=highlight_color)
    connection_patch_kwargs = {'color': 'black',
                               'arrowstyle': '-|>',
                               'mutation_scale': 15,
                               'linewidth': 1,
                               'zorder': 1}
    # ConnectionPatch handles the transform internally so no need to get fig.transFigure
    x0_arrow = patches.ConnectionPatch(
        xyCovariance,
        xyX0,
        coordsA=axis_covariance.transData,
        coordsB=axis_x0.transData,
        **connection_patch_kwargs,
    )
    y_arrow = patches.ConnectionPatch(
        xyCovariance,
        xyY,
        coordsA=axis_covariance.transData,
        coordsB=axis_y.transData,
        **connection_patch_kwargs,
    )
    size_arrow = patches.ConnectionPatch(
        xySize,
        xyCovariance,
        coordsA=axis_size.transData,
        coordsB=axis_covariance.transData,
        **connection_patch_kwargs,
    )
    fig.patches.extend([x0_arrow, y_arrow, size_arrow])
    # Edit "unlikely" marginal bars
    for rect, _ in zip(axis_x0.patches, range(3)):
        rect.set_facecolor(darker_default_color)
    for rect, _ in zip(axis_y.patches, range(2)):
        rect.set_facecolor(darker_default_color)
    axis_size.patches[0].set_facecolor(darker_default_color)
    axis_size.patches[-1].set_facecolor(highlight_hex)
    axis_x0.patches[-1].set_facecolor(lighter_default_color)
    axis_y.patches[-3].set_facecolor(lighter_default_color)

def add_unconditional_lines(fig, axis_x0, axis_y, axis_size, axis_covariance):
    # Add lines between subplots
    xyX0, xyY, xySize, xyCovariance = anchor_points()
    #axis_x0.plot(*xyX0, "o", color=default_color)
    #axis_y.plot(*xyY, "o", color=default_color)
    #axis_size.plot(*xySize, "o", color=default_color)
    axis_covariance.plot([0,1],[0,1], color=highlight_color)
    axis_covariance.plot(*xyCovariance, "o", zorder=2, color=highlight_color)
    connection_patch_kwargs = {'color': 'black',
                               'arrowstyle': '-|>',
                               'mutation_scale': 15,
                               'linewidth': 1,
                               'zorder': 1}
    # ConnectionPatch handles the transform internally so no need to get fig.transFigure
    x0_arrow = patches.ConnectionPatch(
        xyCovariance,
        xyX0,
        coordsA=axis_covariance.transData,
        coordsB=axis_x0.transData,
        **connection_patch_kwargs,
    )
    y_arrow = patches.ConnectionPatch(
        xyCovariance,
        xyY,
        coordsA=axis_covariance.transData,
        coordsB=axis_y.transData,
        **connection_patch_kwargs,
    )
    size_arrow = patches.ConnectionPatch(
        xyCovariance,
        xySize,
        coordsA=axis_covariance.transData,
        coordsB=axis_size.transData,
        **connection_patch_kwargs,
    )
    fig.patches.extend([x0_arrow, y_arrow, size_arrow])
    # Edit rectangle colors
    axis_x0.patches[-2].set_facecolor(lighter_default_color)
    axis_y.patches[-1].set_facecolor(lighter_default_color)
    axis_size.patches[-1].set_facecolor(lighter_default_color)

def main(args):
    # Ensure consistent data
    np.random.seed(1234)
    from gc_vis import source_data, model_covariance

    fig = plt.figure()

    # First subplot: x0
    axis_x0 = fig.add_subplot(221)
    sns.histplot(source_data['x0'], ax=axis_x0, legend=False, stat='proportion')

    # Second subplot: y
    axis_y = fig.add_subplot(222)
    sns.histplot(source_data['y'], ax=axis_y, legend=False, stat='proportion')
    axis_y.yaxis.tick_right()
    axis_y.yaxis.set_label_position('right')

    # Third subplot: size
    axis_size = fig.add_subplot(223)
    sns.histplot(source_data['size'], ax=axis_size, legend=False, stat='proportion')

    # Fourth submplot: covariance
    axis_covariance = fig.add_subplot(224)
    axis_covariance.set_xlabel('Correlation Influence')
    axis_covariance.yaxis.tick_right()
    #mask = np.triu(np.ones_like(model_covariance, dtype=bool))
    #cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Rename index
    #model_covariance = model_covariance.rename(columns={'x0.value': 'x0','y.value': 'y', 'size#1#3.value': 'size'}, index={'x0.value': 'x0', 'y.value': 'y', 'size#1#3.value': 'size'})
    #sns.heatmap(model_covariance, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axis_covariance)

    if args.unconditional:
        add_unconditional_lines(fig, axis_x0, axis_y, axis_size, axis_covariance)
    else:
        add_conditional_lines(fig, axis_x0, axis_y, axis_size, axis_covariance)

    fig.tight_layout()

    # Show figure
    if args.show:
        plt.show()
    else:
        fig.savefig(args.output_path, format=args.output_path.rsplit('.',1)[1], dpi=500)

if __name__ == "__main__":
    main(parse())

