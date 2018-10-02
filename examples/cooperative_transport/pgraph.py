"""Script to draw performance graph for paper."""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from swarms.utils.graph import PGraph


def main():
    filenames = sys.argv[1]
    fdir = sys.argv[2]
    filenames = filenames.split(',')
    # print(fdir)
    # print(filenames)
    graph = PGraph(fdir, filenames)
    graph.gen_plot()


if __name__ == '__main__':
    main()