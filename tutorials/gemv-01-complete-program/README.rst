GEMV 1: A Complete Program
==========================

This example demonstrates a complete CSL program.

A complete program consists of a host program (a Python script, in this example)
and at least two CSL code files,
one of which defines the layout of the program across a collection of
processing elements (PEs) on the Wafer-Scale Engine (hereafter referred to
as "device"),
and one or more of which define the programs running on the individual PEs.
In this example, there is just one PE.

When executing the program, the user first compiles the CSL code files, and
then invokes the host program to copy data on and off the device and launch
functions on the device using a remote procedure call (RPC) mechanism.
The device used may be an actual CS system,
or it may be simulated without access to an actual CS system using the
Cerebras Fabric Simulator.

The host program here is defined in the ``run.py`` script, and the layout and
device code are defined in ``layout.csl`` and ``pe_program.csl``.

The movement of data from host to device and back is done with memory to memory
copy semantics, which is provided by an SDK utility called ``memcpy``.
The top of the ``layout.csl`` file imports a module which is used to
parameterize the program's ``memcpy`` infrastructure.
This file also includes a layout block which specifies the number
and spatial arrangement of PEs used by this program, as well as the instructions
to execute on each PE.
Here, we instruct the compiler to produce executable code for 1 PE using the
code in ``pe_program.csl``.

This program executes as follows.
The host code ``run.py`` uses the remote procedure call (RPC) mechanism to
launch a function called ``init_and_compute`` on the device.
This function initializes a 4 x 6 matrix ``A``, stored in row major format,
a 6 x 1 vector ``x``, and a 4 x 1 vector ``b``.
Then, it computes the matrix-vector product of ``Ax + b``
and stores it in ``y``.

Once ``init_and_compute`` finishes on the device,
the host program performs a device-to-host memcpy with
the ``memcpy_d2h`` command to copy back the result stored in ``y``,
and then checks that the answer is correct.
Notice the ``unblock_cmd_stream`` call in ``pe_program.csl`` that occurs
at the end of ``init_and_compute``;
this call allows the device-to-host ``memcpy_d2h`` to proceed.
