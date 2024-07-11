Topic 10: @map Builtin
======================

The ``@map`` builtin can be used to perform custom operations on the data
elements of one or more DSDs. In other words, it is a
*customizable DSD operation* that allows us to go beyond the
:ref:`fixed list <language-builtins-for-dsd-operations>` of
natively supported DSD operations.

This example demonstrates three use-cases of the ``@map`` builtin:

1. In the first use-case, ``@map`` is used to compute the square-root of the
   diagonal elements of a 2D tensor.
2. In the second use-case ``@map`` is used to perform a custom calculation with
   a mix of input DSDs of various kinds (``mem1d_dsd`` and ``fabin_dsd``) and
   scalar values while the result is stored to a ``mem1d_dsd``. It shows how we
   can use arbitrary callbacks combined with a variety of input and output DSDs.
3. Finally, we demonstrate how ``@map`` can be used to compute a reduction like
   the sum of all elements in a tensor.

Without ``@map``, we would have to write explicit loops iterating over each
element involved in these computations. With ``@map`` we can avoid writing such
loops by utilizing the DSD descriptions which specify the loop structure
implicitly. Since DSDs are supported natively by the hardware, using ``@map``
can lead to significant performance gains compared to writing explicit loops.
