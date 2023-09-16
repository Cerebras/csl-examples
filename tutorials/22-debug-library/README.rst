
Debug Library
=============

This example shows a program that uses the tracing mechanism of the
``<debug>`` library to record variable values and compile time strings
as well as timestamps, for inspection by the host code.

This program uses a row of four contiguous PEs.
Two colors, ``red`` (color 0) and ``blue`` (color 1), are used.
On all PEs, the routing associated with these colors receives
from the ``WEST`` and sends down the ``RAMP`` and ``EAST``.
Additionally, for both colors, ``swap_color_x`` is set to ``true``.
Because these colors differ only in their lowest bit, when a
``red`` wavelet comes into a router from ``WEST``, it leaves the
router to the ``EAST`` as a ``blue`` wavelet, and vice versa.

The host code sends four wavelets along the color ``MEMCPYH2D_DATA_1``
into the first PE. The WTT of ``MEMCPYH2D_DATA_1`` forwards this data
to color ``blue``. When a PE receives a ``red`` wavelet, the task
``red_task`` is activated, and when a PE receives a ``blue`` wavelet,
the task ``blue_task`` is activated.

Each PE program contains a global variable named ``global``,
initialized to zero.
When a ``red_task`` is activated by an incoming wavelet ``in_data``,
``global`` is incremented by an amount ``in_data``.
When a ``blue_task`` is activated by an incoming wavelet ``in_data``,
``global`` is incremented by an amount ``2 * in_data``.

The programs running on each PE import two instances of the
``<debug>`` library. Each time a task activates, the instance
named ``trace`` logs a compile time string noting the color
of the task, and the updated value of ``global``.
The instance named ``times`` logs a timestamp at the beginning
of a task, and at the end of a task.

The host code uses the function ``read_trace`` from
``cerebras.sdk.debug.debug_util`` to read the logged
values after execution of the device code finishes.
Note that the PE coordinates passed to ``read_trace`` start
from the northwest corner of the fabric, not from the
northwest corner of the program rectangle.
