CSL Examples
=====

This repository contains examples of CSL code. Each example has the following properties:

* The source code is complete and self-contained and can be compiled using the CSL compiler.
* The compiled code can be simulated using our fabric simulator, or it can be executed on the Cerebras hardware itself.

Each example is located in its own sub-folder, and contains files of the following types:

* `\*.csl` : CSL source code files. If there is more than one CSL source file, the top-level file (specified when compiling) 
  is named either `layout.csl` or `code.csl`, and it will import all other CSL files.
* `run.py`: This script drives the simulator (or the Cerebras fabric itself). It creates input data, runs the simulator, gets the simulation
  result and compares to an expected result, also computed in this file.
* `commands.sh`: Shell script which contains the exact commands you need to execute to first compile the source code and then simulate it.
* `\*.rst`: Documentation. 

The examples are divided into three main categories, introduction, tutorials and examples, described in the following sections.

Introduction
---------
The `introduction` folder is the place to start. It contains five examples which serves as a first introduction to the core concepts
of the CSL language. We recommend working through these examples in the following order:

#. first-example
#. tensors-and-multicore
#. routes-and-wavelets
#. data-structure-descriptors
#. fabric-switches

The above examples are documented on the `SDK Documentation website <https://sdk.cerebras.net>`_ and not in `\*.rst` files.

Tutorials
------

The material in the `tutorials` folder assumes that you have gone through the introduction first. There are 21 tutorial examples, 
each of which illustrates a specific language feature in more detail. You can work through the examples in the
order suggested by the prefix number of each example. However, you can also work through these examples in almost any other
order and just refer to other tutorials as needed, depending on which concepts are being used.

Examples
---------

The material in the `examples` folder should be studied last. It contains six sample applications, each of which solves a specific problem:

* `gemv`: This is arguably the simples application and therefore a good place to start. It implements generalized matrix-vector multiplication in about 100 lines of CSL.
* `wide-multiplication`: Implements multiplication of two 128-bit unsigned integers in approximately 100 lines of CSL.
* `FFT`: Implements 1D and 2D Discrete Fourier Transforms, DFT in 400+ lines of CSL.
* `residual`: Computes the norm of the residual of  a matrix-vector multiplication. Builds on GEMV. Approximately 500 lines of CSL.
* `histogram-torus`: A communication demo. The fabric memory is filled with random values which are then sorted into buckets, 
  where each bucket is a single processing element of the WSE.
* `stencil`: A 3D 25-point stencil implemented in a little less than 700 lines of CSL.














