Release Notes
=============

The following are the release notes for the CSL Examples repository,
``csl-examples``.

Version 1.4.0
-------------

- The examples are improved and updated to comply with the SDK version 1.3.0.

- A series of new example tutorial programs have been introduced to demonstrate
  the new ``SdkLayout`` API for program layout specification, which is
  currently in beta.

Version 1.3.0
-------------

- The examples are improved and updated to comply with the SDK version 1.3.0.

- A new example program ``row-col-broadcast`` has been introduced which
  benchmarks the bandwidth of data transfers between host and device,
  where data is broadcast across a row or column of PEs,
  using the new ``memcpy_h2d_colbcast`` and ``memcpy_h2d_rowbcast`` APIs.

- A new example program ``game-of-life`` has been introduced which implements
  Conway's Game of Life, where each PE is treated as a single cell.

Version 1.2.0
-------------

- The examples are improved and updated to comply with the SDK version 1.2.0.

- All tutorial example programs have been updated to support WSE-3.

- Two new example programs for switches, demonstrating use of the
  ``<control>`` library, have been added.

- A new example program demonstrating the ``<simprint>`` library has been
  added.

- ``wide-multiplication``, ``residual``, ``mandelbrot``,
  ``gemv-collectives_2d``, ``gemv-checkerboard-pattern``,
  ``gemm-collectives_2d``, ``stencil-3d-7pts``, ``bicgstab``,
  ``conjugateGradient``, ``preconditionedConjugateGradient``, and
  ``powerMethod`` programs have been updated to support WSE-3.

Version 1.1.0
-------------

- The examples are improved and updated to comply with the SDK version 1.1.0.

- GEMV tutorials 1 through 8 have been updated to support WSE-2 and WSE-3.

- ``cholesky``, ``FFT``, ``bandwidthTest``, and ``single-tile-matvec``
  programs have been updated to support WSE-2 and WSE-3.

- New example program introduced to demonstrate WSE-3 features for
  separation of queue IDs from microthread IDs for asynchronous operations.

Version 1.0.0
-------------

- The examples are improved and updated to comply with the SDK version 1.0.0.

- Several more tutorials in the GEMV series have been introduced.

- A tutorial series which builds up an increasingly complex
  pipelined computation model have been introduced.

Version 0.9.0
-------------

- The examples are improved and updated to comply with the SDK version 0.9.0.

- The ``CSELFRunner`` tutorials and benchmarks have been removed. Future
  programs should move to using ``SdkRuntime``.

- In addition to the ``SdkRuntime`` GEMV tutorials, straight ports of the
  legacy ``CSELFRunner`` tutorials have been introduced.

- All remaining ``CSELFRunner`` benchmark programs have been ported to
  ``SdkRuntime``, including ``histogram-torus``, ``mandelbrot``, ``cholesky``,
  ``gemv-collectives_2d``, and ``wide-multiplication``.

Version 0.8.0
-------------

- The examples are improved and updated to comply with the SDK version 0.7.0.

- A new set of ``SdkRuntime`` tutorials have been introduced, which
  introduce language features by building up an increasingly complex
  implementation of GEMV.

- Several new ``SdkRuntime`` benchmark programs have been added, including
  ``spmv-hypersparse``, ``stencil-3d-7pt``, ``powerMethod``,
  ``conjugateGradient``, ``preconditionedConjugateGradient``, and
  ``bicgstab``.

Version 0.7.0
-------------

- The examples are improved and updated to comply with the SDK version 0.7.0.

- The examples have been reorganized to partition them by their host
  runtime, ``CSELFRunner`` or ``SdkRuntime``.

- Five of the tutorial examples now have ``SdkRuntime`` versions.

- ``gemm-collectives_2d`` now has an ``SdkRuntime`` version.

- A new benchmark, ``bandwidthTest``, has been added. There is only
  a version using ``SdkRuntime``.

Version 0.6.0
-------------

- The examples are improved and updated to comply with the SDK version 0.6.0.

- Several new examples added:

    - ``cholesky``
    - ``hadamard-product``
    - ``gemv-collectives_2d``
    - ``gemm-collectives_2d``

- The ``gemv`` example has been refactored and renamed to
  ``gemv-checkerboard-pattern``.

Version 0.5.1
-------------

- The examples are improved and updated to comply with the SDK version 0.5.1.

- Several new examples added:

    - ``histogram-inject``
    - ``mandelbrot``
    - ``residual-memcpy``
    - ``stencil-memcpy``

Version 0.4.0
-------------

- This is the first release of the CSL Examples repository. 

- All examples are 100% compatible with the
  `0.4.0 release of the SDK <https://sdk.cerebras.net>`_.

- Please refer to the ``README.rst`` file for information on the general
  repository structure.
