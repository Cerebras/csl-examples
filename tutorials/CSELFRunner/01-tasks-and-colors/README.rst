
Tasks
=====

This example demonstrates a complete program.  It defines a task, which is a
function that executes in response to color activations.  The ``main_task`` task
updates the global variable to the value ``42``.

Since tasks run in response to color activations, this example defines a color
``main_color``.  Colors are numbered and depending on their value, they enable
specific functionality in the hardware.  This example uses a routable color
value, which is a value in the range 0 through 23.  Subsequent examples
demonstrate other color values.

The code includes a ``comptime`` block, which is a sequence of statements that
execute during program compilation.  See :ref:`language-comptime`
for details.  ``comptime`` statements are roughly analogous to C++ ``constexpr``
expressions, although with some important differences.

In this ``comptime`` block, we tell the compiler to:

- associate the color ``main_color`` with the task ``main_task``
- set the initial state of ``main_task`` to Active, so that the task scheduler
  can schedule the task for execution

The net effect of this ``comptime`` block is that the task ``main_task``
executes as soon as the program is loaded into on-chip memory.

Finally, the ``layout`` block tells the compiler the number of Processing
Elements (PEs) used by this program and the instructions to execute on each PE.
Here, we instruct the compiler to produce executable code for 1 PE using
instructions from the current source file.
