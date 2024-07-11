Topic 14: Color Swap
====================

This example demonstrates the color swap feature of WSE-2.
CSL currently does not support color swap on WSE-3, and support
is in development.

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

Each PE program contains a global variable named ``sum``,
initialized to zero.
When a ``red_task`` is activated by an incoming wavelet ``in_data``,
``sum`` is incremented by an amount ``in_data``.
When a ``blue_task`` is activated by an incoming wavelet ``in_data``,
``sum`` is incremented by an amount ``2 * in_data``.
