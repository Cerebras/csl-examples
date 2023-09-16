GEMV with Collective Communications
===================================

This example shows a CSL program that uses collective communications
to perform a generalized matrix-vector (GEMV)
multiplication operation of the form:

.. code-block:: text

    y = Ax + b

where:

- ``A`` is a tensor of shape [M, N] (stored distributed on PE memory).
- ``x`` is a tensor of shape [N, 1].
  It is placed in the memory of the northwesternmost PE before computation
  begins, and then scattered using collective communications.
- ``b`` is a tensor of shape [M, 1].
  It is placed in the memory of the northwesternmost PE before computation
  begins, and then scattered using collective communications.
- ``y`` is the output tensor of shape [M, 1].
  At the end of computation, it is located in the memory of
  the southeasternmost PE.

For simplicity, we choose N as a multiple of the
width of the kernel and M as a multiple of the height of the kernel.
With the default compile parameters for this example,
M = 32, N = 16 and we use a PE rectangle of size 4×4 for the kernel.
The parameters specifying these values can be modified at compile time.

Note that this algorithm and the implementation are not optimized for
performance. It is intended to serve as a non-trivial introductory example
of the collectives library.

All computations are done in FP32 format.

The matrix ``A``, of shape [M, N],
is distributed across the PE memories as follows:

- The first dimension of ``A``, M rows, is distributed across
  the height of the kernel.
- The second dimension of ``A``, N columns, is distributed across
  the width of the kernel.

Since we know that M is 32 and the height of the kernel is 4, each PE will be
assigned 32÷4 = 8 rows of ``A``.

Similarly, each PE will get 16÷4 = 4 columns of ``A``. This means each PE is
assigned an 8×4 chunk of the original matrix ``A``.
