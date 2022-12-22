Cholesky
--------

If we have a symmetric positive-definite matrix ``A``, then we can use
Cholesky decomposition to find a lower-triangular matrix ``L`` such that
``LL^T = A``.

This benchmark implements the Cholesky decomposition algorithm using the
"right-looking" approach. We can write out the algorithm as:

.. code-block::

  N = A.shape[0]
  for i in range(N):
    pivot = A[i,i]
    A[:,i] /= sqrt(pivot)
    A[i+1:,i+1:] -= np.outer(A[i+1:,i], A[i+1:,i])

Note that because ``A`` is symmetric, we need only store its lower triangle,
and indeed do the whole computation on the lower triangle.

To implement this in CSL, we tile ``A`` over the lower triangle of our grid of
PEs. We run routes with color ``row_color`` across the rows of PEs and routes
with color ``col_color`` down the columns of PEs. We can visualize our triangle
of PEs as follows:

.. code-block::

    ┌───┐
  ┌─┤P00│
  │ └───┘
  │
  │   ┌───────┐
  │   │       │
  │ ┌─┴─┐   ┌─▼─┐
  ├─►P01│ ┌─┤P11│
  │ └───┘ │ └───┘
  │       │
  │   ┌─► │ ──┬───────┐
  │   │   │   │       │
  │ ┌─┴─┐ │ ┌─▼─┐   ┌─▼─┐
  ├─►P02│ ├─►P12│ ┌─┤P22│
  │ └───┘ │ └───┘ │ └───┘
  │       │       │
  │   ┌─► │ ──┬─► │ ──┬───────┐
  │   │   │   │   │   │       │
  │ ┌─┴─┐ │ ┌─▼─┐ │ ┌─▼─┐   ┌─▼─┐
  ├─►P03│ ├─►P13│ ├─►P23│ ┌─┤P33│
  │ └───┘ │ └───┘ │ └───┘ │ └───┘
  │       │       │       │
  │   ┌─► │ ──┬─► │ ──┬─► │ ──┬───────┐
  │   │   │   │   │   │   │   │       │
  │ ┌─┴─┐ │ ┌─▼─┐ │ ┌─▼─┐ │ ┌─▼─┐   ┌─▼─┐
  └─►P04│ └─►P14│ └─►P24│ └─►P34│   │P44│
    └───┘   └───┘   └───┘   └───┘   └───┘

For clarity, each PE stores a an ``Nt x Nt`` sized tile of A. For PEs on the
diagonal, only the lower triangle of the tile is actually stored.

Recall from the code above that the algorithm will need to run for ``N``
iterations. Let's look at what happens in a given outer-loop iteration:

  1. The top left PE (P00) computes the inverse square root of the pivot, and
  multiplies that value by the first column of its tile. It then sends its
  first column down the row color.

  2. PEs along the left column receive P00's chunk of the first column and use
  it to update their first column (multiply by ``invsqrt``). Then, they compute
  an outer product of this updated first column with the chunk received from
  P00. Finally, they send their updated first columns ``EAST`` along the
  ``row_color``.

  3. When row tile reaches PEs along the diagonal (P11, P22, P33, P44), those
  PEs
  subtract an outer product of that row chunk with itself from their own tile's
  values. They then send their received row chunk (unmodified) down the
  ``col_color``

  4. Interior PEs (P12, P13, P23, P14, P24, P34) receive a row chunk along the
  ``row_color`` and a column chunk along the ``col_color``. They subtract
  the outer product of these chunks from their local tiles.

We can then move onto the next outer loop iteration.

The interesting transition
happens once we have done ``Nt`` iterations. At this point, the left-most
column of PEs no longer participates in the algorithm, and column 1 becomes
the next left-most column. P11 assumes the "top left" role previously held by
P00. Importantly, P12, P13, P14 now need to *send* on ``row_col`` instead of
receiving. This means that we need to reconfigure some routes!

Fortunately, this can be achieved using fabric switches on ``row_color``. After
they have finished processing their tile's final column, PEs P02, P03, and
P04 send control wavelets to flip their neighbors' fabric switches to allow
them to send on ``row_color``. Note that P01 does not need to do this because
P11 will never send values on ``row_color``.

This process will repeat again as column 2, then column 3, then finally column 4
become the left-most columns. For the last ``Nt`` many iterations, all PEs other
than P44 will be idle.
