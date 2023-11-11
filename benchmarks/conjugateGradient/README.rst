Conjugate Gradient
==================

This example evaluates the performance of Conjugate Gradient (CG) with a 
sparse matrix ``A`` built by 7-point stencil. The kernel records the ``start``
and ``end`` of CG by tsc counter. In addition the tsc counters of all PEs are
not synchronized in the beginning. To avoid the timing variation among those
PEs, ``sync()`` synchronizes all PEs and samples the reference clock.

There are two implementations, ``kernel.csl`` and ``kernel_cg.csl`` compiled
by ``run.py`` and ``run_cg.py`` respectively. Both kernels define host-callable
functions ``f_sync()``, ``f_tic()`` and ``f_toc()`` in order to synchronize the
PEs and record the timing.

The kernel ``kernel.csl`` also defines a couple of host-callable functions to
implement CG algorithm, including

- ``spmv()``: compute ``A*x`` and ``A*p``

- ``dot()``: compute ``dot(p,w)`` and ``dot(r,r)``

- others: update ``x``, ``p``, ``r``

The kernel ``kernel_cg.csl`` defines a host-callable function ``f_cg`` which
implements the CG on the WSE. The ``f_cg`` introduces a state machine to call a
sequence of ``spmv()``, ``dot()`` and others. Such state machine simply realizes
the algorithm in ``run.py``.

The kernel ``allreduce/pe.csl`` performs a reduction over the whole rectangle
to synchronize the PEs, then the bottom-right PE sends a signal to other PEs
to sample the reference clock.

The kernel ``stencil_3d_7pts/pe.csl`` performs a matrix-vector product (spmv)
where the matrix has 7 diagonals corresponding to 7 point stencil. The stencil
coefficients can vary per PE, but must be the same for the local vector. The
user can change the coefficients based on the boundary condition or curvilinear
coordinate transformation.

The script ``run.py`` or ``run_cg.py`` has the following parameters:

- ``-k=<int>`` specifies the maximum size of local vector.

- ``--zDim=<int>`` specifies how many elements per PE are computed.

- ``--max-ite=<int>`` specifies number of iterations in power method.

- ``--channels=<int>`` specifies the number of I/O channels, no bigger than 16.

The ``tic()`` samples "time_start" and ``toc()`` samples "time_end". The
``sync()`` samples "time_ref" which is used to adjust "time_start" and
"time_end". The elapsed time (unit: cycles) is measured by
``cycles_send = max(time_end) - min(time_start)``

The overall runtime (us) is computed via the following formula
``time_send = (cycles_send / 0.85) * 1.e-3 us``
