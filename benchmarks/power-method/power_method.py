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
from numpy import linalg as LA


def power_method(A_csr, x0, max_ite):
  prev_mu = 0
  nrm2_x = LA.norm(x0, 2)
  x = x0 / nrm2_x
  for i in range(max_ite):
    y = A_csr.dot(x)
    mu = np.dot(x, y)
    print(f"i = {i}, mu = {mu}, |prev_mu - mu| = {abs(mu - prev_mu)}")
    nrm2_x = LA.norm(y, 2)
    x = y / nrm2_x
    prev_mu = mu
  return x
