Topic 3: Streaming Wavelet Data
===============================

Often, CSL programs contain tasks that are activated in response to the
arrival of wavelets of specific colors. Such tasks are also called
Wavelet-Triggered Tasks, or data tasks.

In this example, the ``comptime`` block binds a data task to a ``data_task_id``
created from a ``memcpy`` streaming color, which receives data from the host.
The routing of the color ``MEMCPYH2D_DATA_1`` must not be defined.
The ``memcpy`` module will figure out the routing of ``MEMCPYH2D_DATA_1``.

Given the task and color association and the route, when a wavelet of
color ``MEMCPYH2D_DATA_1`` arrives at the router, it is forwarded to the CE,
which then activates ``main_task``.  The wavelet's payload field is received in
the argument to the task, and the code uses the wavelet data to update a global
variable.
