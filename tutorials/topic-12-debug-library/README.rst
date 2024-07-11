Topic 12: Debug Library
=======================

This example shows a program that uses the tracing mechanism of the ``<debug>``
library to record variable values and compile time strings as well as
timestamps, for inspection by the host code.

The program uses a row of four contiguous PEs.
The first PE sends an array of values to three receiver PEs.
Each PE program contains a global variable named ``global``, initialized to
zero.
When the data task ``recv_task`` on the receiver PE is activated by an incoming
wavelet ``in_data``, ``global`` is incremented by ``2 * in_data``.

The programs running on each PE import two instances of the ``<debug>`` library.
On the receiver PEs, each time a task activates, the instance named ``trace``
logs a compile time string noting that the task has begun execution, and the
updated value of ``global``.
The instance named ``times`` logs a timestamp at the beginning of a task, and
at the end of a task.

The host code uses the function ``read_trace`` from
``cerebras.sdk.debug.debug_util`` to read the logged values after execution of
the device code finishes.
Note that the PE coordinates passed to ``read_trace`` start from the northwest
corner of the fabric, not from the northwest corner of the program rectangle.
