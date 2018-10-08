"""Script to draw performance graph for paper."""

import sys
from swarms.utils.graph import PGraph, BoxGraph


def main():
    """Parse args and call graph module."""
    filenames = sys.argv[1]
    fdir = sys.argv[2]
    filenames = filenames.split(',')
    #graph = PGraph(fdir, filenames, "Single-Source Foraging")
    graph = PGraph(fdir, filenames, "Nest Maintenance")
    graph.gen_plot()

    #box = BoxGraph(fdir, filenames, "Single-Source Foraging")
    box = BoxGraph(fdir, filenames, "Nest Maintenance")
    box.gen_plot()


if __name__ == '__main__':
    main()
