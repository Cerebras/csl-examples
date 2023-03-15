.. _benchmark-gemm-collectives:

GEMM with Collective Operations
===============================

This program implements the SUMMA matrix multiplication algorithm and serves
as an example of using the ``collectives_2d`` library.

We perform GEMM in ``P`` many steps on a grid of ``P x P`` processors.
At each step ``i``, PEs in the ith column broadcast their home tiles of ``A``
to other PEs in their row, and PEs in the ith row broadcast their home
tiles of ``B`` to other PEs in their column. Once both broadcasts are complete
as determined by ``x_done()`` and ``y_done()`` both being activated,
each PE computes ``C_tile += Ap * Bp`` where ``Ap`` and ``Bp`` are pointers to
either the PE's home tile or the tile it received through broadcasts.
