"""Script to draw performance graph for paper."""

import sys
from swarms.utils.graph import PGraph, BoxGraph


def main():
    """Parse args and call graph module."""
    filenames = sys.argv[1]
    fdir = sys.argv[2]
    filenames = filenames.split(',')
    graph = PGraph(fdir, filenames)
    graph.gen_plot()

    box = BoxGraph(fdir, filenames)
    box.gen_plot()


if __name__ == '__main__':
    main()
