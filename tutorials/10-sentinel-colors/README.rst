
Sentinel Colors
===============

In previous programs, we used so-called routable colors, which are colors that
are associated with a route to direct the flow of wavelets.  Routable colors are
in the range 0 through 23.  This example demonstrates the use of non-routable
color to signal the end of a input tensor, thus giving it the name *Sentinel
Color*.

In this example, the host sends the number of wavelets via RPC. The kernel
counts received wavelets via H2D, and trigger the sentinel color when all
wavelets are received.  Since sentinel colors are not routable, the programmer
should not specify a route for them, but they do need to bind the sentinel
color to a task.

Here, the sentinel color activates the ``send_result`` task, which relays the
result of the sum reduction back to the host.
