# import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np


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

  def show_lines(self, xlabels, yvalues, xaxislabel='', yaxislabel='', title=''):
    return
    fig, ax = plt.subplots()

    opacity = 0.8

    colors = ['g', 'r', 'b', 'c', 'm']
    markers = ['o']
    styles = [':', '--', '-.', '-']
    for i, xlabel in enumerate(xlabels):
      yvals = []
      # map boolean to numbers
      for val in yvalues[i]:
        yvals.append(10 if val else 5)

      c = random.sample(colors, 1)[0]
      m = random.sample(markers, 1)[0]
      s = random.sample(styles, 1)[0]
      plt.plot(np.arange(len(yvals)), yvals, c + m + s, linewidth=1,
               markersize=4, alpha=0.5, label=xlabel)

    plt.grid()

    # rects = ax.bar(xlabel, yvalues[i], bar_width,
    # alpha=opacity, color=random.sample(colors, 1), label=xlabel)

    plt.ylabel(yaxislabel)
    plt.legend(loc='upper right')
    # ax.set_xticks(np.arange(0, len(xlabels)))
    # ax.set_xticklabels(xlabels)
    plt.tight_layout()
    plt.show()
