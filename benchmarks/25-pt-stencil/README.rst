25-Point Stencil
================

The stencil code is a time-marching app, requiring the following three inputs:

- scalar ``iterations``: number of time steps
- tensor ``vp``: velocity field
- tensor ``source``: source term

and producing the following three outputs:

- maximum and minimum value of vector field of last time step, two f32 per PE
- timestamps of the time-marching per PE, three uint32 per PE
- vector field ``z`` of last time step, ``zdim`` f32 per PE

The stencil code uses 21 colors and task IDs for communication patterns,
and ``SdkRuntime`` reserves 6 colors,
so only 4 colors are left for ``streaming`` H2D/D2H transfers
and some entrypoints for control flow.
We use one color (color 0) to launch kernel functions
and one entrypoint (color 2) to trigger the time marching.
The ``copy`` mode of memcpy is used for two inputs and two outputs.

After the simulator (or WSE) has been launched,
we send input tensors ``vp`` and ``source`` to the device via ``copy`` mode.

Second, we launch time marching with the argument ``iterations``.

In this example, we have two kernel launches.
One performs time marching after ``vp`` and ``source`` are received,
and the other prepares the output data ``zValues``.
The former has the function symbol ``f_activate_comp``
and the latter has the function symbol ``f_prepare_zout``.
Here ``SdkRuntime.launch()`` triggers a host-callable function, in which
the first argument is the function symbol ``f_activate_comp``,
and the second argument is ``iterations``,
which is received as an argument by ``f_activate_comp``.

The end of time marching (``f_checkpoint()`` in ``task.csl``)
will record the maximum and minimum value
of the vector field and timing info into an array ``d2h_buf_f32``.
The host calls ``memcpy_d2h()`` to receive the data in ``d2h_buf_f32``.

To receive the vector field of the last time step,
the function ``f_prepare_zout()`` is called by ``SdkRuntime.launch()``
to prepare this data into a temporary array ``zout``,
because the result is in either ``zValues[0, :]`` or ``zValues[1, :]``.

The last operation, ``memcpy_d2h()``, sends the array ``zout`` back to the host.

When ``f_activate_comp`` is launched, it triggers the entrypoint ``f_comp()``
to start the time-marching and to record the starting time.

At the end of time marching, the function ``epilog()`` checks
``iterationCount``.
If it reaches the given ``iterations``,  ``epilog()`` triggers the entrypoint
``CHECKPOINT`` to prepare the data for the first ``memcpy_d2h()``.

The function ``f_checkpoint()`` calls ``unblock_cmd_stream()`` to process the
next operation which is the first ``memcpy_d2h()``.
Without ``unblock_cmd_stream()``, the program stalls because the
``memcpy_d2h()`` is never scheduled.

The function ``f_prepare_zout()`` prepares the vector field into ``zout``.
It also calls ``unblock_cmd_stream()`` to process the next operation, which is
the second ``memcpy_d2h()``.
