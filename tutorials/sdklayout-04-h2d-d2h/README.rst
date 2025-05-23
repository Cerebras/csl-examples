.. _tutorials-sdklayout-04:

SdkLayout 4: Host-to-device and device-to-host data streaming
=============================================================

This tutorial demonstrates how we can connect ports to the
host to allow us to stream data in and out of the WSE.

It uses the 'add2vec' code region that was also used in
tutorial :ref:`sdkruntime-sdklayout-03-ports-and-connections` but instead of
using sender/receiver code regions it creates streams directly
to/from the host.

Similar to connections between input and output ports (see tutorial
:ref:`sdkruntime-sdklayout-03-ports-and-connections`) paths to/from ports
to/from the edge of the wafer are produced automatically.

For now, it is only possible to create input/output streams
to/from single-PE ports. If a port consists of more than one PE then
an adaptor layer must be created explicitly to funnel the data
through a single PE port. The next tutorial shows an example
of such a configuration.
