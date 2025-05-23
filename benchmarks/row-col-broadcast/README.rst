Host-to-Device Broadcast Test
=============================

This example shows how to use row or column broadcast. For example if the user
wants to broadcast a column of data [1.0, 2.0, 3.0, 4.0] to a region of interest
starting from (1,1) with width 3 and height 4, one element per PE, the H2D API
requires the user to prepare the following 3-by-4 tensor,

.. code-block::

   | 1.0  1.0  1.0 |
   | 2.0  2.0  2.0 |
   | 3.0  3.0  3.0 |
   | 4.0  4.0  4.0 |

and use ``memcpy_h2d()`` API to stream 12 elements into the device. This
operation wastes host bandwidth by 3x.
Now the user can use the new API, ``memcpy_h2d_rowbcast()``, to stream 4
elements only.

The same for column broadcasting, the user only needs to provide data of one
row and uses ``memcpy_h2d_colbcast()`` API.

The new broadcasting scheme only supports H2D, not D2H.

The kernel of ``row-col-broadcast`` is the same as ``bandwidth-test``.
The ``run.py`` calculates the bandwidth as well.
The formula of the bandwidth calculation is the same as ``bandwidth-test``,
so the user can see how much time this new API can save.
