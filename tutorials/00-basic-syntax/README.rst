
Basic CSL Syntax
================

This example serves to illustrate the syntax of some of the core language
constructs.  The code in this example is not a complete program, but it shows
some of the most commonly-used features of the language.

The language syntax is similar, although not identical, to
`Zig <https://ziglang.org>`_.  However, despite the similarity in the language
constructs, both the purpose as well as the implementation of the CSL
compiler are substantially different from that of the Zig compiler.

Types
-----

CSL includes some basic types such as:


* ``bool`` for boolean values
* ``i16`` and ``i32`` for 16- and 32-bit signed integers
* ``u16`` and ``u32`` for 16- and 32-bit *unsigned* integers
* ``f16`` and ``f32`` for 16- and 32-bit IEEE-754 floating point numbers

In addition to the above, CSL also supports array types and pointer types.
Their use will be illustrated in subsequent examples.

Functions
---------

Functions are declared using the ``fn`` keyword.  The compiler provides special
functions called *Builtins*, whose names start with ``@`` and whose implementation
is provided by the compiler.  All CSL builtins are described
`here <../../Language/Builtins.rst>`_.

Conditional Statements and Loops
--------------------------------

CSL includes support for ``if`` statements and ``while`` and ``for`` loops.  These
are described in greater detail in the subsequent examples.
