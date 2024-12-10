Conway's Game of Life
=====================

This program implements
`Conway's Game of Life <https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life>`_
on the WSE.

Conway's Game of Life is a cellular automaton which evolves on a 2D grid of
square cells. Each cell is in one of two possible states, LIVE or DEAD.
Every cell interacts with its neighbors, which are the cells horziontally,
vertically, or diagonally adjacent. At each step in time, the following
transitions occur:

- Any LIVE cell with fewer than two LIVE neighbours becomes a DEAD cell.
- Any LIVE cell with two or three LIVE neighbours stays a LIVE cell.
- Any LIVE cell with more than three LIVE neighbours becomes a DEAD cell.
- Any DEAD cell with exactly three LIVE neighbours becomes a LIVE cell.

This program implements the Game of Life be assigning one cell to each PE.
Zero boundary conditions are used, and thus the neighbors of a border PE that
fall outside of the program rectangle are treaded as always DEAD.

In each generation, each PE sends its state to its four N, S, E, and W
neighbors. Each PE receives the state of its four N, S, E, and W neighbors, and
also forwards the received state from its N and S neighbors to its E and W
neighbors. Thus, each PE receives from its E and W links both the state of its
E and W adjacent neighbors, as well as its four diagonal neighbors.

The program implements two initial conditions, ``random`` and ``glider``.
``random`` randomly initializes the state of all cells. ``glider`` generates
several glider objects across the grid. The initial condition can be set with
the ``--initial-state`` flag.

The ``--show-ascii-animation`` flag will generate an ASCII animation of the
cellular automoton's evolution when the program is complete.
``--save-animation`` will save a GIF of the automoton's evolution.
