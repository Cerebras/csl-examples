Topic 6: Switches
=================

Fabric switches permit limited runtime control of routes.

In this example, the ``layout`` block initializes the default route to receive
wavelets from the ramp and forward them to the PE's north neighbor.  However, it
also defines routes for switch positions 1, 2, and 3.  The hardware updates the
route according to the specified switch positions when it receives a so-called
Control Wavelet.

For the payload of the control wavelet, the code creates a special wavelet using
the helper function ``encode_single_payload()`` from the ``<control>`` library.
The program then sends out a data wavelet along the newly-switched color.

Switches can be helpful not just to change the routing configuration in limited
ways at runtime, but also to save the number of colors used.  For instance, this
same example could be re-written to use four colors and four routes, but by
using fabric switches, this example uses just one color.
