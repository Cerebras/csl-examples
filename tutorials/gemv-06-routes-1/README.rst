GEMV 6: Routes and Fabric DSDs, Part I
======================================

Continuing from the previous example, we now break up a single GEMV
computation among two PEs.

The host program copies ``b`` into the ``y`` tensor of the left PE.
The left PE also gets the first ``N/2`` columns of ``A`` and the first ``N/2``
values of ``x``, and the right PE gets the last ``N/2`` columns of ``A``
and last ``N/2`` values of ``x``.

The left and right PE both increment their local ``y`` tensors by computing
their piece of ``Ax``.
Then, the left PE sends its result to the right PE, which increments its ``y``
tensor by the received values.

Last, the host copies ``y`` from the right PE, and checks that the result is
correct.

To send data from the left PE to the right PE, we must specify a route, known
as a color.
In ``layout.csl``, ``@set_color_config`` specifies that on the left PE,
color 0 will receive data, or wavelets, from the compute element (CE)
up the RAMP, and transmit them to the EAST.
On the right PE, color 0 will receive wavelets form the ``WEST``, and then
transmit them down the RAMP to the CE.
``@set_tile_code`` passes the ID of this color to ``pe_program`` as a
parameter named ``send_color``, and also sets a paremeter called ``pe_id``,
to diffentiate if the program is running on the left or the right PE.

The ``send_right`` function executed on the left PE defines a ``fabout_dsd``
called ``out_dsd`` that sends ``M`` wavelets along the color route specified
by ``send_color``.
``out_dsd`` is used as the destination operand of ``@fmovs``, and ``y_dsd``
as the source operand.
Thus, this operation sends the ``M`` elements accessed by ``y_dsd`` along the
fabric as specified by ``out_dsd``.

The ``recv_left`` function executed on the right PE receives the data in a
``fabin_dsd`` named ``in_dsd``, used in an ``@fadds`` operation that
increments the ``M`` elements of ``y`` on this PE by the ``M`` received values.

Note that this program also provides an example of a local task.
The ``@fmovs`` and ``@fadds`` operations are performed asynchronously;
when these operations are done, the color ``exit_color`` is activated, which
activates the task ``exit_task``.
This task unblocks ``memcpy``'s command stream, allowing additional commands
from the host program to proceed.
