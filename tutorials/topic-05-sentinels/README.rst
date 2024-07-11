Topic 5: Sentinels
==================

In previous programs, we used so-called routable colors, which are associated
with a route to direct the flow of wavelets.
On WSE-2, task IDs which can receive data wavelets are in the range 0 through
23, corresponding to the IDs of the colors.
On WSE-3, task IDs which can receive data wavelets are in the range 0 through
7, corresponding to input queues which are bound to a routable color.
We have also used local tasks, which on WSE-2 can be associated with any task
ID from 0 to 30, and on WSE-3 can be associated with any task ID from 8 to 30.

This example demonstrates the use of a non-routable control task ID to signal
the end of an input tensor.
We call this use for a control task ID a *sentinel*.

In this example, the host sends to a receiving PE (``sentinel.csl``) the number
of wavelets that the receiving PE should expect to receive, followed by the
stream of data.
The receiving PE then sends the data to its neighbor (``pe_program.csl``),
followed by a *control wavelet* which specifies the control task ID that the
neighbor will activate.

Since sentinel control task IDs are not routable colors, the programmer does
not specify a route, but does need to bind the control task ID to a control
task, which will be activated upon receipt of the sentinel wavelet.
Here, the sentinel activates the ``send_result`` task, which relays the
result of the sum reduction back to the host.
