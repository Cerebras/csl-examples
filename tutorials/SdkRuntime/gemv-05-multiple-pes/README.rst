GEMV 5: Multiple PEs
====================

Continuing on from the previous example, we now extend our program to use
multiple PEs.

The number of PEs used in this program is set at compile-time with the ``width``
parameter.
Note that ``layout.csl`` uses this parameter to set the size of the program
with the call to ``@set_rectangle``.
The dimensions of a grid of PEs is always specified as width by height (or,
alternatively, number of columns by number of rows), and individual PEs are
indexed by (x, y), or, in other words, (column number, row number).

This program involves no communication between PEs; we only duplicate the same
workload on each PE.
In ``run.py``, the ``memcpy_h2d`` calls now specify that data is copied into
``width x 1`` PEs, beginning at the upper left corner (0, 0) of the program
rectangle.
Because we are copying the same data to each PE, we use ``np.tile`` to repeat
the data in ``A``, ``x``, and ``b`` multiple times.
The ``memcpy_d2h`` call copies back the resulting ``y`` from each PE into
an array of size ``M x width``.

The next example will expand this example to demonstrate simple communication
between PEs.
