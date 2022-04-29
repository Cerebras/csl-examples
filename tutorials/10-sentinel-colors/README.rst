
Sentinel Colors
===============

In previous programs, we used so-called routable colors, which are colors that
are associated with a route to direct the flow of wavelets.  Routable colors are
in the range 0 through 23.  This example demonstrates the use of non-routable
color to signal the end of a input tensor, thus giving it the name *Sentinel
Color*.

In this example, the host sends a sentinel color wavelet at the end of the
wavelets for the input tensor.  Since sentinel colors are not routable, the
programmer should not specify a route for them, but they do need to bind the
sentinel color to a task, which will be activated upon receipt of the sentinel
color wavelet.

Here, the sentinel color activates the ``send_result`` task, which relays the
result of the sum reduction back to the host.
