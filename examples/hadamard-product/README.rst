Hadamard Product
================

This program demonstrates selective batch mode in the ``CSELFRunner`` runtime,
which allows for a list of input and output tensors to be specified for distinct
execution phases of a running program.

The program calculates the Hadamard Product (per-PE element-wise product)
of two input tensors, returning the result in an output tensor,
when a batch is run with ``inColorA``, ``inColorB``, and ``outColorA``.
These colors correspond to, respectively, the color along which the first
input tensor is copied, the color along which the second input tensor is copied,
and the color along which the output Hadamard product is copied.
After each batch run with these inputs, the output is verified by comparing
to the Hadamard product as computed on the host.

Additionally, the very first batch is run with ``inColorC``, which copies
an integer value ``report_count`` to each PE.
During each batch run that computes a Hadamard product,
this value is decremented. When ``report_count``'s value reaches 0, the value
:math:`2^{\text{iter count}}` is copied back from each PE when a batch is run
with ``outColorB``.
