Single Tile Matvec
==================

This program performs single tile matrix-vector products y = A*x,
where A has dimension N x N, and the data type is ``fp32``.
This program produces memory bandwidth information and FLOPS.

To compile and run for N=10 to N=100 on an actual CS-2, use:
``./sweep.py --dims 750,994 --cmaddr <CS IP ADDR>``

Dims here refers to the dimensions of the program rectangle.
The program above will perform 750*994 matvecs, with one matvec
occurring on each PE, for each value of N.

There is also an ``iters`` flag, which allows you to average
cycle counts over multiple runs. Here is an example for a 
10 x 10 program rectangle run in the simulator:
``./sweep.py --dims 10,10 --iters 10``.

You can also compile and run separately, and also run in a verificaiton
mode that verifies the result on each PE.

To compile:

.. code-block::

   cslc layout_matvec.csl --fabric-dims=17,12 --fabric-offsets=4,1 \
   --params=width:10,height:10,tile_size:25,iters:1 -o out --memcpy --channels=1

where ``fabric-dims`` is the dimension of the simfabric, ``width``, ``height``
are the dimensions of the program rectangle, ``tile_size`` is N,
and ``iters`` is the number of iterations over which we average.

Note that the width must be no bigger than 7 less than the x-dimension of the fabric,
and the height must be no bigger than 2 less than the y-dimension of the fabric.

Additionally, if you are running on a real CS-2, ``fabric-dims`` must
be ``750,994``.

To run:

.. code-block::

   cs_python run.py --name out --verify

where the ``--verify`` flag verifies the result on each PE.
Note that this flag is incompatible with any number of iterations greater than 1.

Again, pass ``--cmaddr`` to run on a real CS-2.
If you're just running on simfabric, there's no need to use a width or height
greater than one. The cycle counts will be the same across all simulated PEs.

The program will report a "relative" and "absolute" memory bandwidth. Relative
refers to the bandwidth obtained if we calculate based on the number of memory
accesses that would occur to implement a matrix-vector product on a more
traditional architecture, where caching allows us to write the y vector
for example, only once to memory. "Absolute" refers to the actual number of
memory accesses required to run this program on the WSE. See comments in the
code for more details.
