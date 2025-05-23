.. _tutorials-sdklayout-03:

SdkLayout 3: Ports and connections
==================================

This tutorial demonstrates how to attach ports to code regions
and then connect those ports together. It instantiates two
code regions that send data to a third code region. The receiving
code region adds the input streams element-wise and then sends
the result out and towards a fourth code region that saves the
result on device memory.

There are two kinds of ports: input ports and output ports. It is
only possible to connect an output port to an input port. When
we do that the ``SdkLayout`` compiler will automatically find and
encode a path between them.
