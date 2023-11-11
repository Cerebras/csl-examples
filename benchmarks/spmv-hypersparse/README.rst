spmv-hypersparse
================

This example evaluates the performance of sparse matrix-vector multiplication.
The kernel records the ``start`` and ``end`` of ``spmv`` by tsc counter. In
addition the tsc counters of all PEs are not sychronized in the beginning.
To avoid the timing variation among those PEs, ``f_sync()`` synchronizes all
PEs and samples the reference clock.

The kernel ``kernel.csl`` defines a couple of host-callable functions,
``f_sync()``, ``f_tic()`` and ``f_toc()`` in order to synchronize the PEs and
record the timing of ``spmv``.

The kernel ``allreduce2R1E/pe.csl`` performs a reduction over the whole rectangle
to synchronize the PEs, then the bottom-right PE sends a signal to other PEs
to sample the reference clock. The ``allreduce2R1E`` is a variant of ``allreduce``
in ``stencil-3d-7pts``. The former uses 2 routable colors and 1 entrypoints, the
latter uses 1 routable color and 4 entrypoints. ``allreduce2R1E`` is designed for
spmv kernel which only has three unused colors. 

The kernel ``hypersparse_spmv/pe.csl`` performs a matrix-vector product (spmv)
where the matrix ``A`` is hypersparse, partitioned into 2D grids. The input 
vector ``x`` and output vector ``y`` are also distributed into 2D grids.

The user has to provide the matrix ``A`` in Matrix Market File format with 1-based
index. To obtain the best performance, the user may need to reorder the matrix
such that the variatoin of the nonzeros of each parition is small. One option is
``util/analyze.cpp`` which provides a load balancing algorithm.

The script ``run.py`` has the following parameters:

- ``--infile_mtx=<path to mtx file>`` contains the sparse matrix A

- ``--num_pe_rows=<int>`` specifies the height of the core rectangle

- ``--num_pe_cols=<int>`` specifies the width of the core rectangle

- ``--channels=<int>`` specifies the number of I/O channels, no bigger than 16.

The ``tic()`` samples "time_start" and ``toc()`` samples "time_end". The
``sync()`` samples "time_ref" which is used to adjust "time_start" and
"time_end". The elapsed time (unit: cycles) is measured by
``cycles_send = max(time_end) - min(time_start)``

The overall runtime (us) is computed via the following formula
``time_send = (cycles_send / 0.85) * 1.e-3 us``

The bandwidth is calculated by
``bandwidth = ((2*nnz+m)*4)/time_send)``
