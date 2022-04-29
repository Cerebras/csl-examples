
Routes
======

Until now, we have been activating the task's color at compile time, but
typically, CSL programs contain tasks that are activated in response to the
arrival of wavelets of specific colors.  Such tasks are also called
Wavelet-Triggered Tasks.

In this example, the ``comptime`` block binds a task to a color, but it doesn't
activate it.  Additionally, the ``layout`` block defines a route for wavelets of
the color ``main_color`` to be directed from the west connection of the PE to the
ramp, thus forwarding them to the Compute Element (CE).

Given the task and color assocation and the above route, when a wavelet of color
``main_color`` arrives at the router, it is forwarded to the CE, which then
activates ``main_task``.  The wavelet's payload field is received in the argument
to the the task, and the code uses the wavelet data to update a global variable.
