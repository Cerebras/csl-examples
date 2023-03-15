
Multiple Source Files
=====================

This example demonstrates a clean separation between instructions that execute
on the PEs and instructions to the compiler for generating code for various PEs.
In general, we recommend such a separation to make the code easier to read.

The top-level source file (``code.csl``) defines the ``layout`` block, which
instructs the compiler to generate code for 1 PE whose instructions are located
in the file ``pe_program.csl``.  The path to any files mentioned in the
``@set_tile_code()`` builtin is relative to the location of the top-level source
file.

In addition to specifying the source file which contains program instructions,
this example also parameterizes the code in the referenced file with a color
value.

This program assigns the value 42 to a variable, and then sends this value
to the EAST on an output wavelet.
