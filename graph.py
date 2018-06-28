# import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np

##################
# This class encapsulates graph related functions
##################


class Grapher(object):

  def show_bar(self, xlabels, yvalues, xaxislabel='', yaxislabel='', title=''):
    fig, ax = plt.subplots()

    bar_width = 0.75
    opacity = 0.8

    colors = ['b', 'g', 'r', 'c', 'm']

    selected_colors = random.sample(colors, len(xlabels))
    for i, xlabel in enumerate(xlabels):
      rects = ax.bar(xlabel, yvalues[i], bar_width,
                     alpha=opacity, color=selected_colors[i % len(xlabels)], label=xlabel)

    ax.set_xlabel(xaxislabel)
    ax.set_ylabel(yaxislabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(0, len(xlabels)))
    ax.set_xticklabels(xlabels)
    fig.tight_layout()
    plt.show()

  def show_lines(self, predictions_map, xaxislabel='', yaxislabel='', title=''):
    '''
    Shows the delta between predicted and actual using dots

    '''
    fig, ax = plt.subplots()

    opacity = 0.8

    colors = ['g', 'r', 'b', 'c', 'm']
    markers = ['o']
    styles = [':', '--', '-.', '-']

    # compare 2 arrays for their differences
    diff = predictions_map['actual'] == predictions_map['predicted']

    # change values to integer for plotting
    match_val = 0
    non_match_val = 1
    diff_to_integer = [match_val if val else non_match_val for val in diff]

    c = random.sample(colors, 1)[0]
    m = random.sample(markers, 1)[0]
    s = random.sample(styles, 1)[0]

    ax.semilogx(np.arange(len(diff_to_integer)),
                diff_to_integer, c + m, markersize=2, alpha=0.8, label='top dots show prediction mismatch')

    plt.text(2, 0.5, 'total errors {0}'.format((diff == False).sum()),
             horizontalalignment='center',
             verticalalignment='top',
             multialignment='center')

    plt.xlabel(xaxislabel + '  total: ({0})'.format(len(diff_to_integer)))
    plt.ylabel(yaxislabel)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()
