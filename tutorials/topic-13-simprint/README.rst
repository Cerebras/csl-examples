Topic 13: Simprint Library
==========================

When running with the simulator, you can also print values directly to the
simulator logs (``sim.log``).
This example modifies the previous example to show the use of the
``<simprint>`` library for printing comptime strings and values to the
simulator log.

Just like the previous example, this program uses a row of four contiguous PEs.
The first PE sends an array of values to three receiver PEs.
Each PE program contains a global variable named ``global``, initialized to
zero.
When the data task ``recv_task`` on the receiver PE is activated by an incoming
wavelet ``in_data``, ``global`` is incremented by an amount ``2 * in_data``.

On the receiver PEs, each time a task activates, the program writes to
``sim.log`` a string denoting that the task has started, along with the value
of the wavelet received, and the updated value of ``global``.
The program also defines a helper function ``simprint_pe_coords`` to print out
the coordinates of the PE to the simulator log.
The output is flushed to ``sim.log`` whenever a newline is encountered, so you
must explicitly print ``"\n"`` to flush the output.

After running this example, open up ``sim.log`` to see the output.
The output from ``<simprint>`` should look something like this:

.. code-block::

  @968 PE(0,0): sender beginning main_fn
  @996 PE(0,0): sender exiting
  @1156 PE(1,0): recv_task: in_data = 0, global = 0
  @1158 PE(2,0): recv_task: in_data = 0, global = 0
  @1160 PE(3,0): recv_task: in_data = 0, global = 0
  @1338 PE(1,0): recv_task: in_data = 1, global = 2
  @1340 PE(2,0): recv_task: in_data = 1, global = 2
  @1342 PE(3,0): recv_task: in_data = 1, global = 2
  @1520 PE(1,0): recv_task: in_data = 2, global = 6
  @1522 PE(2,0): recv_task: in_data = 2, global = 6
  @1524 PE(3,0): recv_task: in_data = 2, global = 6
  @1702 PE(1,0): recv_task: in_data = 3, global = 12
  @1704 PE(2,0): recv_task: in_data = 3, global = 12
  @1706 PE(3,0): recv_task: in_data = 3, global = 12
  @1884 PE(1,0): recv_task: in_data = 4, global = 20
  @1886 PE(2,0): recv_task: in_data = 4, global = 20
  @1888 PE(3,0): recv_task: in_data = 4, global = 20

Note that each line printed to ``sim.log`` is prepended with the cycle at which
the print is encountered.

``<simprint>`` is particularly useful for debugging stalling programs.
The ``<debug>`` library shown in the previous example requires a program to
complete to parse its output, but the ``<simprint>`` library prints to
``sim.log`` whenever a newline character is encountered.
