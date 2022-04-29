
Memory Data Structure Definition
================================

The Cerebras architecture includes instructions that repeat certain operations
using Data Structure Definitions (DSDs).  This example demonstrates the use of
such instructions over memory addresses.

This example creates three one-dimensional memory DSDs, each of which tells the
hardware how to loop over a different array.  The ``tensor_access`` field
specifies an induction variable, a loop bound, and an affine expression to
generate various addresses at runtime.

The code then uses these three DSDs in a DSD operation (``@faddh``) to store the
results of a point-wise addition of two arrays into a third array.  Other DSD
operations and their associated operand types are described
`here <../../Language/Builtins.rst#builtins-for-dsd-operations>`_.
