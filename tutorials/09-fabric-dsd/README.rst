
Fabric Data Structure Definitions
=================================

While wavelet-triggered tasks enable us to receive and operate on one wavelet at
a time, the programmer may need a way to receive a tensor comprised of multiple
wavelets using one instruction.  This is enabled by fabric input DSDs.
Similarly, using fabric output DSDs, the programmer can send multiple wavelets
using one instruction.

This example illustrates two fabric DSDs, one for input and another for output.
Each fabric DSD requires a corresponding color and a corresponding route.

Crucially, when using a fabric input DSD, it is important that the programmer
block the wavelet's color, as this example does for the color ``inColor``.
Otherwise, wavelets of that color will attempt to activate the (empty) task
associated with the color, which in turn will consume the wavelet before it can
be consumed by the fabric input DSD.
