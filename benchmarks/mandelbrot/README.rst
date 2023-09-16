Mandelbrot
==========

This is a simple program that computes a visualization of the Mandelbrot set on
a 16x16 pixel grid using a 4x4 grid of PEs.

Files:
- ``code.csl``: the main file that sets up the 4x4 PE grid and routing
- ``left.csl``: code for the PEs on the left of the grid
- ``middle.csl``: code for the PEs in the rest of the grid
- ``common.csl``: Mandelbrot code used in all PEs

Description:

This program adopts a pipeline-parallel approach to generating the Mandelbrot
set. Each row of 4 PEs is responsible for a 4x16 chunk of the grid. The PE on
the left of each row generates elements, performs up to 8 iterations on them,
then passes them to the right. Each subsequent PE in the same row will also
perform up to 8 iterations, then pass the elements right. Eventually, the
element is outputted on the EAST side of the grid after having undergone a
maximum of 32 iterations.

When a PE passes "an element", it is actually passing 3 32-bit floats. They are
as follows: { real part, imaginary part, number of iterations so far }

Middle PEs calculate the x,y of the values they receive based on the order they
receive them in.

An alternative approach would be to assign each PE a 4x4 tile of the 16x16
overall grid and have it compute Mandelbrot for just its tile. Implementing this
version and comparing its performance to this pipeline-parallel program would be
interesting future work.

Known problems:
- Load balancing between PEs in the same row is poor
- ``iters`` is stored as an ``f32``. It really should be an integer type,
however, we do not yet have support for sending structs through memory DSDs.
