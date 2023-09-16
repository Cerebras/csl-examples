Wide Multiplication
===================

This example shows a CSL program that performs multiplication of wide integers:

.. code-block:: text

    result = X x Y

where:

- ``X`` and ``Y`` are 128-bit unsigned integers.
- ``result`` is the 256-bit wide result of multiplying X and Y.

The simulation script ``run.py`` generates random values for ``X`` and ``Y``.
``X`` is represented as a NumPy array of 16 elements of type ``uint16`` on the
form

.. code-block::

    X = [x₀, x₁, ..., x₇, 0, 0, ..., 0]

where:

- The representation uses little endian.
- x :subscript:`i`, i = 0, 1,..., 7, is the i-th 2-byte word of ``X``.
- The eight trailing zeros are leading zeros to get a full 256-bit
  representation.

``Y`` is represented similarly, and ``X`` and ``Y`` are concatenated and sent to
the fabric as a single 32-element vector of type ``uint16``:

.. code-block::

     (X, Y) = [x₀, x₁, ..., x₇, 0, 0, ..., 0, y₀, y₁, ... y₇, 0, 0,
               ..., 0]

The multiplication is performed by a single PE which receives the input vectors
(``X``, ``Y``) via the streaming H2D on color ``MEMCPYH2D_DATA_1`` and delivers
the ``result`` via streaming D2H on color ``MEMCPYD2H_DATA_1``. A single color
``MEMCPYH2D_DATA_1`` is used for the delivery of both input vectors ``X`` and
``Y``. This is made possible by concatenation of ``X`` and ``Y`` into a single
input vector.

The multiplication is done at the bit level. In the k'th iteration of the outer
loop, ``Y`` is traversed and multiplied by the bit value at position ``k`` of
``X``. This partial result is added to an accumulated result, tracking a carry
bit, and ``X`` is then shifted by one position before the next iteration.
