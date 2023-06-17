GEMV 4: Parameters
==================

Parameter values are compile-time constants, which implies that the compiler
is fully aware of their precise value.
This enables the programmer to not just change the program’s behavior at
runtime, but it also enables the programmer to change the program’s
compilation.

Continuing on from the previous example, we add two compile-time parameters
to the ``layout.csl`` file that specify the dimensions ``M`` and ``N`` of our
problem, instead of hardcoding them in ``pe_program.csl``.
When the program is compiled, the program specifies ``M`` and ``N`` in the
compile command. ``layout.csl`` also sets these parameter  values in
``pe_program.csl`` in its ``@set_tile_code`` call.
