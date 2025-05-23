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


# solve a linear system A * x = b
# where A is a unsymmetric matrix
#
# The algorithm is modified from
#  H. A. VAN DER VORST, BI-CGSTAB: A FAST AND SMOOTHLY CONVERGING VARIANT
#  OF BI-CG FOR THE SOLUTION OF NONSYMMETRIC LINEAR SYSTEMS,
#  SIAM J. ScI. STAT. COMPUT. Vol. 13, No. 2, pp. 631-644, March 1992
#
# Input
#  A_csr     sparse matrix of type scipy.sparse.csr_matrix
#  x0        initial guess, could be a random vector or the approximated solution
#            of some other iterative solver
#  b         right-hand-side vector
#  max_ite   maximum number of iterations
#  tol       tolerance to stop the algorithm
#            the bigger, the more iterations
#            usually tol = eps * |b| where eps > 1.e-6 for f32
# Output
#  x         approximated solution of A*x=b
#  xi        |b - A*x|^2
#  k         the number of iterations
#
# BiCGSTAB does not converge, it hits serious breakdown when
# rho = (r0, rj) is close to zero.
# Such behavior is confirmed by scipy.sparse.linalg.bicgstab
def bicgstab(A_csr, x0, b, max_ite, tol):
  # storage
  #  b: right-hand-side, read-nly
  #  r0: read-only
  #  x: initial value x0
  #  r: r = b-A*x
  #  p
  #  v
  #  s: can align with r
  #  t
  # To save more space, the input b can be a temporary buffer r
  # Exit: x0 contains the final result, b contains the residual
  k = 0
  # x = x0
  x = np.copy(x0)
  # r0 = b - A*x0
  y = A_csr.dot(x)
  r0 = b - y
  # r = r0
  r = np.copy(r0)
  # p = r0
  p = np.copy(r)
  # xi = |r0|^2
  xi = np.dot(r, r)
  # rho = (r0, r0)
  rho = xi
  print(f"[bicgstab] iter {k}: xi = {xi}")
  # if |r_k|_2 < tol, then exit
  while (xi > tol * tol) and (k < max_ite):
    k = k + 1
    # v = A*p
    v = A_csr.dot(p)
    # alpha = rho / (r0, v)
    r0_dot_v = np.dot(r0, v)
    alpha = rho / r0_dot_v
    # s = r - alpha*v
    s = r - alpha * v
    # t = A*s
    t = A_csr.dot(s)
    # w = (t,s)/(t,t)
    t_dot_s = np.dot(t, s)
    t_dot_t = np.dot(t, t)
    w = t_dot_s / t_dot_t
    # x = x + alpha*p + w*s
    x = x + alpha * p + w * s
    # r = s - w*t
    r = s - w * t
    # update rho
    rho_old = rho
    # rho = (r0, r)
    rho = np.dot(r0, r)
    # beta = (rho/rho_old)*(alpha/w)
    beta = (rho / rho_old) * (alpha / w)
    # p = r + beta*(p - w*v)
    p = r + beta * (p - w * v)
    # update the residual
    xi = np.dot(r, r)
    # serious breakdown when rho = (rj, r0) is close to zero
    # if such scenario happens, we need to restart
    print(f"[bicgstab] iter {k}: xi = {xi}, rho = {rho}")
    if abs(rho) < 1.0e-4:
      print(f"WARNING: breakdown due to tiny rho ({rho}), restart")
      break
  return x, xi, k
