.. _tutorials-sdklayout-02:

SdkLayout 2: Basic routing
==========================

This tutorial demonstrates how to define routes between the
PEs of a code region using symbolic colors.

The key point here is that the colors that we use for the routes
are symbolic (i.e., without a physical values). This means that
the ``SdkLayout`` compiler will assign the value automatically.

For debugging purposes, the ``SdkLayout`` compiler will emit
a JSON file called ``colors.json`` that contains the allocated
physical color values.
