
Fabric Data Structure Definitions
=================================

While wavelet-triggered tasks enable us to receive and operate on one wavelet at
a time, the programmer may need a way to receive a tensor comprised of multiple
wavelets using one instruction.  This is enabled by fabric input DSDs.
Similarly, using fabric output DSDs, the programmer can send multiple wavelets
using one instruction.

This example illustrates two fabric DSDs, one for input and another for output.
Each fabric DSD requires a corresponding color.

Crucially, when using a fabric input DSD, it is important that the programmer
blocks the wavelet's color, as this example does for the color
``MEMCPYH2D_DATA_1``.
Otherwise, wavelets of that color will attempt to activate the (empty) task
associated with the color, which in turn will consume the wavelet before it can
be consumed by the fabric input DSD.

This example only has a single PE, which receives data via H2D and sends out it
via D2H in one vector operation. Logically speaking it is NOT valid because H2D
and D2H are serialized. The host triggers D2H only if H2D is done. The HW has
some internal queues to hold the data for I/O, so H2D finishes when it pushes
all data into the dedicated queues. This example still works if the size does
not exceed the capacity of such queues. Otherwise H2D stalls.
