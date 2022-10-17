
HISTOGRAM
=========

This example shows a CSL program that creates a histogram of a distributed input
vector.

Histogram takes the following parameters:

- ``INPUT_SIZE`` is the number of input elements per PE.
- ``NUM_BUCKETS`` is the number buckets present on each PE.
- ``BUCKET_SIZE`` is the size of each bucket.
- ``HIST_WIDTH`` is the width of the histogram kernel in PEs.
- ``HIST_HEIGHT`` is the height of the histogram kernel in PEs

Bucket Distribution
-------------------

Each PE contains ``NUM_BUCKETS`` buckets of size ``BUCKET_SIZE`` each. The
buckets are distributed statically across all the PEs in a row-major order. For
example, given a kernel of size 4x4, a ``BUCKET_SIZE`` of 10, and
``NUM_BUCKETS`` = 5, we have the following bucket distribution:

::

  -----------------------------------------------------
  | [0, 50)    | [50, 100)  | [100, 150) | [150, 200) |
  |------------|------------|------------|------------|
  | [200, 250) | [250, 300) | [300, 350) | [350, 400) |
  |------------|------------|------------|------------|
  | [400, 450) | [450, 500) | [500, 550) | [550, 600) |
  |------------|------------|------------|------------|
  | [600, 650) | [650, 700) | [700, 750) | [750, 800) |
  -----------------------------------------------------


Per-tile code
-------------

Each tile performs the following actions for each input data point:

#. Perform the required arithmetic to calculate the target PE for that data
   point.
#. Encode a wavelet that contains the target PE information.
#. When the target ROW is reached, but not yet the correct COL,
#. it re-routes the wavelet to the W/E
#. If both the target row and column IDs correspond to those of the current PE,
   increment the appropriate bucket count locally.

In the meantime, each PE handles traffic from both directions. Each wavelet is
processed by each PE along the route, where it checks whether the wavelet
belongs to it, in which case it keeps it, otherwise, it passes it along.
Wavelets traveling in the north/south direction may need to be rerouted
east/west, if the incoming wavelet matches the PE's row ID, but not column ID.

The wavelet encoding is as follows:

- 10 bits for the Y ID.
- 10 bits for the X ID.
- 12 bits for the bucket ID.

::

  |---10(y)---|---10(x)---|----12(bucket)----|

Once the wavelet arrives at the destination row (or if it already originated
from the target row), the encoding changes to the following:

- 10 bits for the X ID.
- 10 empty bits.
- 12 bits for the bucket ID.

::

  |---10(x)---|-----10----|----12(bucket)----|

This would make it easier to extract the X ID without performing any masking.

Termination criteria
--------------------

No single PE knows when all the other PEs are done with all their work. Thus, we
use a special column of PEs that collects completion signals from all the core
histogram PEs. We call the row the Tally kernel.

Each histogram PE keeps a count of how many wavelets it has put in a bucket.
A "tally" PE at the end of each row polls and aggregates the counts.
The last tally PE aggregates the counts from the other tally PEs. Once the count
reaches the total count (the total number of inputs), it signals off the fabric.


TODOs
=====

Time collection with tally
--------------------------

The cycle count of the program should use as a end_cycle the moment when tally
sends the output wavelet, which marks the end of the kernel. The tally kernel
should be augmented to consider this.

CE Inject coordinates
-----------------------

As a workaround, while running on CS2 we would need to run the bogus while loop
inside task init, since we should guarantee that the registers are written
before any PE starts sending wavelets

Simulator ignoring coordinates
-------------------------------

The simulator is ignoring the call where the X/Y coordinates are set at runtime
@bitcast
The simulator seems to consider by default that every PE has the coordinates
determined by their position in the fabric wrt to the matching process of CE
Inject. CS2 considers all zeros by default, so this needs to be written.
For simplicity, we setup in CS2 what the simulator considers by default, but
once this bug is fixed in the simulator, we could set PE_X and PE_Y instead of
+1, to make comparisons easier throughout the CSL code of SPMV.

Faster Execution
-----------------

To accelerate the code one could replace the software-queues (as an array and
head/tail pointers) with the native FIFOs of the WSE.
Many condition-checking instructions of fifo full/empty could be removed then.
