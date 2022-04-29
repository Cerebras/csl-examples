
Initializers
============

In CSL, all declared variables, global or otherwise, have to be initialized.
Such initializations can be cumbersome when declaring large arrays, so the
compiler provides two builtins ``@zeros()`` and ``@constants()`` for generating
tensors of various dimensions and sizes.

The builtin ``@zeros()`` accepts a numeric array type and produces an array of
zeros.

The builtin ``@constants()`` accepts not just an array type, but also a constant
value, and the builtin returns an array of the specified dimensions and size,
wherein each element is initialized to the provided constant value.
