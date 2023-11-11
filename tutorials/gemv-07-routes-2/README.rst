GEMV 7: Routes and Fabric DSDs, Part II
=======================================

Continuing from the previous example, we now break up a single GEMV
computation among a 2 x 2 square of PEs.

The host program copies ``b`` into the ``y`` tensor of the left column of PEs,
with PE (0, 0) getting the first ``M/2`` values and PE (0, 1) getting the
last ``M/2`` values.

Each PE also gets a corresponding chunk of ``A``.
The left PEs get the left ``N/2`` columns and the right PEs
get the right ``N/2`` columns,
while the upper PEs get the upper ``M/2`` rows and the lower PEs
get the lower ``M/2`` rows.
In other words, the northwest PE gets the northwest quadrant of ``A``,
the northeast PE gets the northeast quadrant of ``A``, and so on.

The host program also copies ``x`` into the upper row of PEs,
with PE (0, 0) getting the first ``N/2`` values and the PE (1, 0)
gettin the last ``N/2`` values.

When the ``compute`` function is launched, the PEs in the top row begin
sending their respective elements of ``x`` to their routers,
along the color ``x_color``.
These PEs send the elements of ``x`` both to the ``SOUTH`` and back down
their own ``RAMP``.

On all four PEs, receiving a wavelet along ``x_color`` activates
``recv_x``. This task is a wavelet-triggered task (WTT): the wavelet's
data is fed in as an argument to ``recv_x``.

When a PE receives an element of ``x`` in the ``recv_x`` task, it computes
the corresponding piece of ``Ax`` and adds it to its local ``y`` tensor.
When a PE has received all corresponding elements of ``x`` along ``x_color``,
(i.e., the first ``N/2`` values of ``x`` for the two left PEs,
and the last ``N/2`` values of ``x`` for the two right PEs),
it has finished computing its local contribution to ``y``.

At this point, the local task ``reduce`` is activated.
The left column of PEs send their partial ``y`` result along the color
``ax_color`` to the ``EAST``, and the right column of PEs receives these
partial ``y`` results, and increments their ``y`` tensors
by the received values.
At this point, the right column of PEs contain the final result ``y``,
with the first ``M/2`` elements in PE (1, 0)
and the last ``M/2`` elements in PE (1, 1).

Last, the host copies ``y`` from the right column of PEs,
and checks that the result is correct.

Note that in this program's layout file, we no longer assign a ``pe_id``
as a compile-time parameter.
Instead, we use the ``<layout>`` module in ``pe_program.csl``
to determine the coordinates of the PE at runtime.
This can reduce compilation time by reducing the
number of unique PE programs that need to be compiled.
Specifically, by parameterizing a PE's code (i.e., passing
parameters through ``@set_tile_code``) we are creating more
unique PE programs as opposed to relying on
runtime-evaluated values.
