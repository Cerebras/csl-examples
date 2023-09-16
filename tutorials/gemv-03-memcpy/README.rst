GEMV 3: H2D and D2H Memcpy
==========================

The memcpy functionality of ``SdkRuntime`` allows the programmer to copy data
between the host and device.
Continuing from the previous example, we now extend it to include
``memcpy_h2d`` calls which copy data from the host to initialize ``A``, ``x``,
and ``y`` on device.
