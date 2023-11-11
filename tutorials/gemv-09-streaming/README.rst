GEMV 9: Memcpy Streaming Mode
=============================

We present an alternative version of the previous example,
in which we use the ``streaming`` mode of ``memcpy`` to stream ``x`` and ``b``
onto the device, and stream ``y`` off of the device.
All of the previous examples used the ``copy`` mode of ``memcpy``.
This example is meant to simply present the basics of ``streaming`` mode,
and future tutorials will demonstrate some use cases for this mode.

The host code no longer includes an explicit kernel launch.
Instead, computation is started by the wavelet-triggered tasks that receive
elements of ``x`` and ``b`` along the top row and left column of PEs,
respectively.
We finish computation when the kernel streams back the result ``y``
to the host.

The colors ``MEMCPYH2D_DATA_1`` and ``MEMCPYH2D_DATA_2`` are used
to stream ``x`` and ``b`` onto the device, respectively,
while ``MEMCPYD2H_DATA_1`` is used to stream ``y`` off the device.

Note that, because ``memcpy`` commands are serialized, the order of these
``streaming`` mode ``memcpy_h2d`` calls in this example is important.
If the ``b`` values were streamed in before ``x``, the program would hang.
