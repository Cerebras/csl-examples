# Copyright 2022 Cerebras Systems.
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

# solve a linear system A * x = b
# where A is a symmetric positive definite matrix
#
# The conjugate gradient method is adopted from Algorithm 10.2.1 of the book
#  GENE H. GOLUB, CHARLES F. VAN LOAN, MATRIX COMPUTATIONS, THIRD EDITION
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
#  rho       |b - A*x|^2
#  k         the number of iterations
# 
def conjugateGradient(A_csr, x0, b, max_ite, tol):
  k = 0
  x = np.copy(x0)
  # r0 = b - A*x0
  y = A_csr.dot(x)
  r = b - y
  # rho = |r0|^2
  rho = np.dot(r,r)
  print(f"[CG] iter {k}: rho = {rho}")
  # if |r_k|_2 < tol, then exit
  while ( (rho > tol*tol) and (k < max_ite) ):
    k = k + 1
    if k == 1:
      # p1 = r0
      p = r
    else:
      # beta_{k} = |r_{k-1}|^2/|r_{k-2}|^2
      beta = rho/rho_old
      # p_{k} = r_{k-1} + beta_{k} * p_{k-1}
      p = r + beta * p
    # alpha_{k} = |r_{k-1}|^2/<p_{k}, A*p_{k}>
    w = A_csr.dot(p)  # w = A*p_{k}
    eta = np.dot(p,w) # eta = <p_{k}, A*p_{k}>
    alpha = rho/eta
    # x_{k} = x_{k-1} + alpha_{k} * p_{k}
    x = x + alpha * p
    # r_{k} = r_{k-1} - alpha_{k} * A*p_{k}
    r = r - alpha * w
    # update rho 
    rho_old = rho
    rho = np.dot(r,r)
    print(f"[CG] iter {k}: rho = {rho}")
  return x, rho, k
