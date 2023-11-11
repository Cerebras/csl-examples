GEMV 8: Routes and Fabric DSDs, Part III
========================================

Continuing from the previous example, we now extend the GEMV computation
to a ``kernel_x_dim x kernel_y_dim`` rectangle of PEs.
We make one simplification,
enforcing that ``M`` is a multiple of ``kernel_y_dim``,
and ``N`` is a multiple of ``kernel_x_dim``.

The host program copies ``b`` into the ``y`` tensor of the left column of PEs,
with each PE getting the corresponding ``M/kernel_y_dim`` values.
Each PE also gets a corresponding chunk of ``A``,
consisting of ``M/kernel_y_dim x N/kernel_x_dim`` elements.
Similarly, the host program copies ``x`` into the upper row of PEs,
with each PE getting ``N/kernel_x_dim`` values.

When the ``compute`` function is launched, the PEs in the top row begin
sending their respective elements of ``x`` to their routers,
along the color ``x_color``.
These PEs send the elements of ``x`` both to the ``SOUTH`` and back down
their own ``RAMP``.
All other rows receive elements of ``x`` along ``x_color`` from the ``NORTH``
and transmit them down their ``RAMP``, with all but the last row also
transmitting the elements further ``SOUTH``.

On all PEs, receiving a wavelet along ``x_color`` activates
``recv_x``. This task is a wavelet-triggered task (WTT): the wavelet's
data is fed in as an argument to ``recv_x``.

When a PE receives an element of ``x`` in the ``recv_x`` task, it increments
its local ``y`` tensor by computing the corresponding piece of ``Ax``.
When a PE has received all corresponding elements of ``x`` along ``x_color``,
with each PE receiving ``N/kernel_x_dim`` values,
it has finished computing its local contribution to ``y``.

At this point, the local task ``reduce`` is activated.
We use two colors in a *checkerboard pattern* to accumulate the partial
``y`` results from ``WEST`` to ``EAST``.
On the even columns, we use ``ax_color_1`` to receive partial results
from the ``WEST`` and ``ax_color_2`` to send partial results to the ``EAST``;
on the odd columns, we use ``ax_color_2`` to receive partial results
from the ``WEST`` and ``ax_color_1`` to send partial results to the ``EAST``.
We must use this checkerboard pattern because we cannot safely send
and receive multiple wavelets on the same color with a fixed routing.
In a future example, we will demonstrate the use of dynamic switching
to update the color routing, which will allow you to use a single color.

The leftmost column of PEs has nothing to receive from the ``WEST``,
so these PEs only send their partial ``y`` results to the ``EAST``.
The remaining columns, upon receiving a partial ``y`` result from the ``WEST``,
increment their ``y`` tensors by the received values,
and all but the rightmost column sends those values to the ``EAST``
The values in the rightmost column contain the final result ``y``,
with each PE containing ``M/kernel_y_dim`` elements.

Last, the host copies ``y`` from the right column of PEs,
and checks that the result is correct.
