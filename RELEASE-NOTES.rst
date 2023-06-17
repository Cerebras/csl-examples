Release Notes
=============

The following are the release notes for the CSL Examples repository,
``csl-examples``.

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
