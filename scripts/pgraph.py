"""Script to draw performance graph for paper."""

import sys
from swarms.utils.graph import PGraph, BoxGraph


def main():
    """Parse args and call graph module."""
    filenames = sys.argv[1]
    fdir = sys.argv[2]
    filenames = filenames.split(',')

    # print (filenames)
    graph = PGraph(fdir, filenames, "Multi-Source Foraging")
    # graph = PGraph(fdir, filenames, "Cooperative Transport")
    # graph = PGraph(fdir, filenames, "Nest Maintenance")
    # graph = PGraph(
    # fdir, filenames, "Nest Maintenance \n with \n Handcoded behaviors")
    graph.gen_plot()

    # box = BoxGraph(fdir, filenames, "Single-Source Foraging")
    # box = BoxGraph(fdir, filenames, False, (-1, 100), "Multi-Source Foraging")
    box = BoxGraph(fdir, filenames, False, (-1, 120), "Nest Maintenance with Handcoded behaviors")
    # box = BoxGraph(
    # fdir, filenames, "Nest Maintenance \n with \n Handcoded behaviors")
    box.gen_plot()


if __name__ == '__main__':
    main()
