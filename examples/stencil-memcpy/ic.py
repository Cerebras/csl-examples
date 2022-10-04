#!/usr/bin/env cs_python
# This is not a real test, but a module that gets imported in other tests.

import numpy as np


def computeGaussianSource(iterations):
  tau = np.float32(1.0)
  scale = np.float32(8.0)
  mscale = np.float32(-8.0)
  _fmax = np.float32(25.0)
  dt = np.float32(0.001610153)
  sigma = np.float32(0.6) * _fmax

  t = np.arange(0, iterations, 1, dtype=np.float32) * np.float32(dt)
  power = np.power(sigma * t - tau, 2, dtype=np.float32)
  expf = np.exp(np.multiply(power, np.float32(mscale)))
  source = (
      np.float32(-2.0)
      * scale
      * sigma
      * np.multiply(
          sigma - np.float32(2.0) * sigma * scale * power,
          expf,
          dtype=np.float32,
      )
  )

  first_zero_idx = np.nonzero(source)[-1][-1] + 1
  if first_zero_idx < source.shape[-1]:
    source = source[:first_zero_idx]
    sourceLength = first_zero_idx
  else:
    sourceLength = source.shape[-1]

  print(f"sourceLength = {sourceLength}, first_zero_idx={first_zero_idx}")

  return source, sourceLength
