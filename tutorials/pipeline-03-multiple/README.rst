
Pipeline 3: Add an artificial halo
==================================

The disadvantage of FIFO in the previous example is the resource consumption.
The FIFO requires two microthreads and a scratch buffer.

The simple workaround is to move such FIFO outside the kernel. We add another
halo, which we call an artificial halo, around the kernel (``pe_program.csl``).
The west side is ``west.csl`` and east side is ``east.csl``.
The ``west.csl`` implements a FIFO to receive the data from H2D.
The ``east.csl`` implements a FIFO to receive the data from ``pe_program.csl``
and redirect it to D2H.

There is no more FIFO in ``pe_program.csl``. Instead, we replace the colors
``MEMCPYH2D_DATA_1`` by ``Cin`` and ``MEMCPYD2H_DATA_1`` by ``Cout``.
The color ``Cin`` receives data from the west to the ramp.
The color ``Cout`` sends the data from ramp to the east.

This example has the same property as ``pipeline-02-fifo``: as long as the
parameter ``size`` does not exceed the capacity of the FIFO in ``west.csl``,
H2D can always finish so the ``@add16`` can progress.
