SdkLayout 5: Generalized matrix-vector multiplication (GEMV)
============================================================

This tutorial demonstrates how we can put all the pieces together
to write a GEMV program. Specifically, this tutorial implements
the following GEMV formulation using 32-bit IEEE 754 floating point
numbers:

.. code-block:: text

    y = Ax + b

- ``A`` is a tensor of shape [M, N] (stored distributed on PE memory).
- ``x`` is a tensor input of shape [N, 1] (streamed in).
- ``b`` is a tensor input of shape [M, 1] (streamed in).
- ``y`` is the tensor output of shape [M, 1] (streamed out).

For simplicity, we choose M as a multiple of the
height of the kernel and N as a multiple of the width of the kernel.
In this example, M = 32, N = 16 and we use a PE-rectangle (kernel) of
size 4×4.

Below is a visualization of the kernel interface:

.. _fig-gemv-4-by-4-checkerboard:

.. figure:: ./images/gemv-4-by-4.png
    :align: center
    :width: 980 px


Note that this algorithm and the implementation are not optimized for
performance. It is intended to serve as a non-trivial introductory example.

All computations are done in IEEE 754 FP32 format.

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

This tutorial demonstrates a few more SDK+Paint features that are
not covered by other previous tutorials. Most importantly, it shows how
we can stream data to/from the host using I/O ports that consist
of more than a single PE unlike the previous tutorial
(see :ref:`sdkruntime-sdklayout-04-h2d-d2h`) where I/O ports had only 1 PE.

As it is explained in the code comments, in order to achieve this we
introduce a demux layer that demultiplex a single-PE input stream into
the multi-PE input ports for vector 'x' and 'b'. We also introduce a
multiplexing layer that fuses a multi-PE stream back into a single-PE
output stream for the result vector 'y'.
