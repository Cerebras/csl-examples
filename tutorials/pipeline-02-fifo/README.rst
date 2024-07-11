
Pipeline 2: Attach a FIFO to H2D
================================

The previous example stalls if the parameter ``size`` exceeds the capacity of
the internal queues. The size of the queue is architecture-dependent. From the
software development point of view, a program should be independent of any
architecture. One solution is to add a FIFO between H2D and ``@add16``. The FIFO
receives data from H2D and then forwards the data to ``@add16``. The WSE
provides an efficient design for FIFO. The user just binds two microthreads to
the FIFO: one pushes data into the FIFO, and the other pops the data out. As
long as the parameter ``size`` does not exceed the capacity of the FIFO, H2D can
always push all data into the FIFO even if ``@add16`` cannot process any data.
Once H2D is done, D2H can continue to drain the data out such that ``@add16``
can progress.

To create a FIFO, we use a builtin ``@allocate_fifo`` to bind a normal tensor.
We create two fabric DSDs: one pushes data from ``MEMCPYH2D_DATA_1`` to the
FIFO and the other pops data from the FIFO to the color ``C1``. Both DSDs must
use different microthreads.

The routing configuration of color ``C1`` is RAMP to RAMP because
1) the FIFO pops data to the router via ``C1`` and
2) ``@add16`` receives data from the router via ``C1``

The disadvantage of this approach is the resource consumption. The FIFO
requires two microthreads and a scratch buffer.

The next example will fix this issue.
