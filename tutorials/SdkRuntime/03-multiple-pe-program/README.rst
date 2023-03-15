
Multiple PEs
============

This example shows a program that uses a row of four contiguous PEs.  The
``layout`` block in the top-level source file (``code.csl``) specifies the
rectangle to be of width 4 and height 1, and it tells the compiler to use the
instructions in the file ``pe_program.csl`` for generating code for each PE.

This program also illustrates a way to enable each PE to distinguish itself from
others by receiving an integer (``pe_id``) from the layout block.  Parameters
are explained in more detail in the next example.

In this program, each of the four PEs receive an integer ``pe_id``,
increment it by 42, and then streams its result to the host using the
``memcpy`` framework.
