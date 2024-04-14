
CSL implementation of 1D and 2D FFT
===================================

This example implements 1D and 2D Discrete Fourier Transforms, DFT. The
algorithm used is Cooley-Tukey, Decimation in Time (DIT), radix 2, with the
slight tweak that we use iteration instead of recursion.

The 1D and 2D FFT CSL implementations share the following characteristics:

#. An input tensor of random real values, denoted by ``X`` in the code, is
   generated on the host (``run.py`` script) and is preloaded into the PEs.
#. The twiddle factors (tensor of complex values), denoted ``f_twiddle`` in
   the code, are also computed on the host and preloaded into the PEs.
#. The host reads the FFT result (tensor of complex values) from PE memories,
   also stored in ``X``.
#. The above means that there is no data streaming and therefore no halo is
   needed.
#. A tensor of ``N`` complex values is represented as a ``2N`` vector of
   floating point values where the real and imaginary parts of the ``i``'th
   value are at index ``2i`` and ``2i+1``, respectively.
#. Computations can be done with 16- or 32-bit floating point precision


1D FFT implementation
---------------------

The 1D implementation uses a single PE only. The core of the computation is in
the file ``fft.csl``. The index reordering of the input ``X`` tensor, i.e., the
bit-reversed indexing scheme used with the butterfly structure, is implemented
by maintaining an auxiliary tensor which in each major step ensures that all
even-indexed elements are in the lower half and all odd-indexed elements are in
the upper half, i.e., even and odd elements are contiguous.

Since part of the implementation is generalized to three dimensions, the ``X``
tensor of ``N`` complex values are written to (and read from) the fabric as a 3D
tensor of shape ``(x, y, z) = (1, 1, 2N)``, where the inner dimension contains
the ``2N`` real values representing ``N`` complex values.

2D FFT implementation
---------------------

For an input of ``N x N`` complex elements, the 2D implementation uses ``N`` PEs
organized in a single row, i.e., the PE rectangle has dimensions ``Nx1``. Each
PE initially holds a column of ``N`` elements. The algorithm then proceeds in
three main steps:

#. In parallel, each PE computes a 1D FFT on its own column of ``N`` elements.
   We now have an ``N x N`` matrix in which the ``i``\th column, on which a 1D
   FFT has been performed, is held by the PE with index ``i``.
#. The ``N x N`` matrix is transposed. Consider the PE at index ``i`` and its
   column elements ``[0, 1, ..., N-1]``. To transpose this column PE ``i`` needs
   to send its elements ``[0, 1, ..., i-1]`` to the PEs with indices
   ``0, 1, ..., i-1``, one element to each PE. Similarly, PE ``i`` needs to send
   its elements ``[i+1, ...,N-1]`` to the PEs with indices ``i+1, ... N-1``, one
   element to each PE. So a given PE sends one element to each of the other PEs
   and it receives ``N-1`` elements from ``N-1`` other PEs. In the CSL
   implementation, this is achieved while parallelizing the transpose of the
   upper and lower triangular submatrices. When the transposition is completed,
   each PE now holds a row of values on which a 1D FFT has already been
   performed.
#. In parallel, each PE now computes a 1D FFT on its row of data, yielding the
   final 2D FFT result.

Due to the 3D generalized code mentioned above, an input ``X`` tensor of size
``N x N`` is written to (and read from) the fabric as a 3D tensor of shape
``(x, y, z) = (N, 1, 2N)``, where the inner dimension contains the ``2N`` real
values representing N complex values.

Required Input Parameters
-------------------------

* ``DIM``: Valid values are ``1`` or ``2``, indicating 1D FFT or 2D FFT,
  respectively.
* ``Nz``: Problem size, corresponding to ``N`` in the descriptions above. ``Nz``
  must be an integer power of 2. If ``DIM == 1``, then ``Nz >= 2`` is required.
  If ``DIM == 2``, then ``Nz >= 4`` is required.
* ``FP``: Floating point precision. Valid values are ``1`` or ``2``, specifying
  16- or 32-bit precision, respectively.
