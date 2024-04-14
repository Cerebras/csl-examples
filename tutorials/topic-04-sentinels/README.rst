
Topic 4: Sentinels
==================

In previous programs, we used so-called routable colors, which
are associated with a route to direct the flow of wavelets.
On WSE-2, task IDs which can be associated with routable colors
are in the range 0 through 23.
This example demonstrates the use of a non-routable control task ID
to signal the end of an input tensor, thus giving it the name *sentinel*.

In this example, the host sends a sentinel wavelet at the end of the
wavelets for the input tensor. Since sentinel control task IDs are not
routable colors, the programmer should not specify a route for them,
but they do need to bind the control task ID to a control task,
which will be activated upon receipt of the sentinel wavelet.

Here, the sentinel activates the ``send_result`` task, which relays the
result of the sum reduction back to the host.
