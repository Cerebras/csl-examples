
GEMV 0: Basic CSL Syntax
========================

This example is the first in a series of successive example programs
demonstrating CSL and the SDK by implementing a general matrix-vector product,
or GEMV.

We start by illustrating the syntax of some of CSL's core language constructs.
The code in this example is not a complete program, but it shows
some of the most commonly used CSL features.

CSL’s syntax is like that of `Zig <https://ziglang.org>`_.
Despite the similarity, both the purpose and the implementation of the CSL
compiler are different from that of the Zig compiler.

Types
-----

CSL includes some basic types:


* ``bool`` for boolean values
* ``i16`` and ``i32`` for 16- and 32-bit signed integers
* ``u16`` and ``u32`` for 16- and 32-bit unsigned integers
* ``f16`` and ``f32`` for 16- and 32-bit IEEE-754 floating point numbers

In addition to the above, CSL also supports array types and pointer types.
Their use will be illustrated in subsequent examples.

Functions
---------

Functions are declared using the ``fn`` keyword.  The compiler provides special
functions called *Builtins*, whose names start with ``@`` and whose
implementation is provided by the compiler.  All CSL builtins are described in
:ref:`language-builtins`.

Conditional Statements and Loops
--------------------------------

CSL includes ``if`` statements and ``while`` and ``for`` loops.
These are described in greater detail in the subsequent example programs.
