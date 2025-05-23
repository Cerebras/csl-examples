#!/usr/bin/env cs_python

# Copyright 2025 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
