3D FFT
======

This example implements a 3D Discrete Fourier Transform by using a pencil
decomposition, in which the input data is viewed as a 2D array of 1D pencils,
and each PE stores a small subarray of the 2D array of pencils.

The algorithm proceeds in steps. First, the 1D FFT of the pencils on each PE
are performed. Then, the data is transposed along a coordinate axis among
all PEs. This process happens two more times, resulting in three local
operations in which 1D FFTs are performed independently on each PE, and three
transpose operations in which all PEs commmunicate to change which axis of
the data is stored in memory.

The algorithm used to compute the 1D FFTs is Cooley-Tukey,
Decimation in Time (DIT), radix 2, with the
slight tweak that we use iteration instead of recursion.

FFT Compilation Parameters
--------------------------

* ``N``: Size of 3D FFT along one dimension. The full problem size is
  ``N x N x N``.
* ``NUM_PENCILS_PER_DIM``: Number of pencils along a given dimension on each PE.
  For instance, ``NUM_PENCILS_PER_DIM == 2`` means that each PE stores
  ``2 x 2`` pencils.
* ``FP``: Floating point precision. Valid values are ``1`` or ``2``, specifying
  IEEE fp16 or fp32, respectively.

FFT Runtime Parameters
----------------------

* ``--inverse``: With this flag set, perform an inverse Fourier transform.
* ``--real``: With this flag set, compute Fourier transform with real input
  data. Without this flag, complex Fourier transform is computed.
* ``--norm``: Normalization strategy. Valid values are ``0``, ``1``, or ``2``,
  specifying ``forward``, ``backward``, or ``orthonormal``, respectively.
