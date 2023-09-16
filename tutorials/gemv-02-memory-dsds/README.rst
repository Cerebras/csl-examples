GEMV 2: Memory DSDs
===================

Continuing on from the previous example, we now extend it by introducing
memory Data Structure Descriptors (DSDs), an efficient mechanism for
performing operations on entire tensors.

This program creates three one-dimensional memory DSDs for accessing ``A``,
``b``, and ``y``, each of which specifies how to loop over the respective
arrays.
The ``tensor_access`` field specifies an induction variable, a loop bound,
and an affine expression (i.e., a linear function plus a constant) to generate
various addresses at runtime.

``b_dsd`` and ``y_dsd`` access the ``M`` contiguous elements of ``b`` and ``y``,
respectively.
``A_dsd`` accesses ``M`` elements of ``A``, but strided by ``N`` elements.
Because ``A`` is stored in row major format, this means that ``A_dsd``
initially accesses the 0th column of ``A``.

These DSDs are used by the DSD operations ``@fmacs`` and ``@fadds`` to
compute ``Ax + b`` and store it in ``y``.
The ``gemv`` function first loops over ``N``, with the ``@fmacs`` in iteration
``i`` computing the scalar-vector product of ``x[i]`` with column ``i``
of ``A``, and incrementing ``y`` by that result.
The ``increment_dsd_offset`` operation updates ``A_dsd`` by shifting its
access by one element.
This causes ``A_dsd`` to access the next column of ``A``.
After the loop, ``y`` is incremented by ``b`` with the ``@fadds`` operation,
to complete the computation.

Other DSD operations and their associated operand types are described in
:ref:`language-builtins-for-dsd-operations`.
