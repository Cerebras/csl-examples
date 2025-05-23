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


import argparse

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkLayout,
    SdkTarget,
    SdkRuntime,
    SimfabConfig,
    get_platform,
)

from demux import get_demux_adaptor, get_x_demux, get_b_demux
from mux import get_mux
from gemv import get_gemv


def get_random_data(size):
  return np.random.uniform(0.0, 1.0, size).astype(np.float32)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cmaddr', help='IP:port for CS system')
  parser.add_argument(
      '--arch',
      choices=['wse2', 'wse3'],
      default='wse3',
      help='Target WSE architecture (default: wse3)'
  )
  args = parser.parse_args()

  ###########
  ### Layout
  ###########
  # If 'cmaddr' is empty then we create a default simulation layout.
  # If 'cmaddr' is not empty then 'config' and 'target' are ignored.
  config = SimfabConfig(dump_core=True)
  target = SdkTarget.WSE3 if (args.arch == 'wse3') else SdkTarget.WSE2
  platform = get_platform(args.cmaddr, config, target)
  layout = SdkLayout(platform)

  ######################
  ### Common invariants
  ######################
  data_width = 16
  data_height = 32
  pe_width = 4
  pe_height = 4
  batch_width = data_width // pe_width
  batch_height = data_height // pe_height
  control_ep = 40

  #########
  ### Data
  #########
  x = get_random_data(data_width)
  b = get_random_data(data_height)
  A = get_random_data(data_width * data_height)
  y = np.empty(data_height, dtype=np.float32)

  #############
  ### Vector X
  #############
  (x_port, x_adaptor_port, x_adaptor) = get_demux_adaptor(
      layout, 'x_demux_adaptor', batch_width, pe_width
  )
  x_adaptor.place(1, 0)

  (x_demux_port, x_out_port, x_demux) = get_x_demux(
      layout, 'x_demux', batch_width, pe_width, 1, control_ep
  )
  x_demux.place(5, 0)
  layout.connect(x_adaptor_port, x_demux_port)

  #############
  ### Vector b
  #############
  (b_port, b_adaptor_port, b_adaptor) = get_demux_adaptor(
      layout, 'b_demux_adaptor', batch_height, pe_height
  )
  b_adaptor.place(1, 2)

  (b_demux_port, b_out_port, b_demux) = get_b_demux(
      layout, 'b_demux', batch_height, 1, pe_height
  )
  b_demux.place(3, 2)
  layout.connect(b_adaptor_port, b_demux_port)

  #########
  ### GEMV
  #########
  (gemv_x_port, gemv_b_port, gemv_y_port, gemv) = get_gemv(
      layout, 'gemv', pe_width, pe_height, batch_width, batch_height, control_ep, A
  )
  gemv.place(5, 2)

  #############
  ### Vector y
  #############
  (y_in_port, y_port, y_mux) = get_mux(layout, 'y_mux', batch_height, 1, pe_height)
  y_mux.place(10, 2)

  #####################
  ### Port connections
  #####################
  layout.connect(x_out_port, gemv_x_port)
  layout.connect(b_out_port, gemv_b_port)
  layout.connect(gemv_y_port, y_in_port)

  ################
  ### I/O streams
  ################
  x_stream = layout.create_input_stream(x_port)
  b_stream = layout.create_input_stream(b_port)
  y_stream = layout.create_output_stream(y_port)

  ################
  ### Compilation
  ################
  compile_artifacts = layout.compile(out_prefix='out')

  ##############
  ### Execution
  ##############
  runtime = SdkRuntime(compile_artifacts, platform, memcpy_required=False)
  runtime.load()
  runtime.run()

  ##################################
  ### Send 'x' and 'b'. Receive 'y'
  ##################################
  # Vectors 'x' and 'b' must be sent asynchronously (i.e., in a
  # non-blocking fashion) to prevent a deadlock due to their
  # inter-dependence in the core gemv computation kernel.
  runtime.send(x_stream, x, nonblock=True)
  runtime.send(b_stream, b, nonblock=True)
  runtime.receive(y_stream, y, data_height, nonblock=True)

  runtime.stop()

  #################
  ### Verification
  #################
  A_matrix = A.reshape(data_height, data_width)
  expected = np.dot(A_matrix, x) + b
  assert np.allclose(expected, y, atol=1e-6)
  print("SUCCESS!")


if __name__ == '__main__':
  main()
