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

from cerebras.sdk.runtime.sdkruntimepybind import (
    Edge,
    Route,
    RoutingPosition,
)

# The mux code region does the inverse job of the demux code region.
# Specifically, it multiplexes the output wavelets from the gemv's
# multi-PE output port for result vector 'y', into a single-PE
# stream that can then be connected to an output stream towards the
# host.
#
# Each PE in the mux layer (which is positioned vertically in this
# example but it can have any orientation) receives 'batch_size'
# wavelets. However, only the PE associated with the single-PE output
# port at the top is able to forward its wavelets out and towards the
# host.
#
# Once 'batch_size' wavelets are forwarded though, a control wavelet
# is emitted to switch the routing position such that the incoming flow
# of wavelets is now received from the south (i.e., from the rest of
# the PEs in the mux layer) and then forwarded out of the single-PE
# output port towards the host.
#
# With that mechanism, one-by-one each PE forwards its 'batch_size'
# wavelets upwards and towards the output port and eventually to the
# host.
#
#               stream 1st batch                  stream 2nd batch
#                 to the host                       to the host
#                     ^                                  ^
#                     |                                  |
#                 +---|----+                         +---|----+
#                 |   |    |                         |   |    |
#  host data--------->o    |          host data      |   o    |
# (batch_size)    |        |         (batch_size)    |   ^    |
#                 |---^----|                         |---|----|
#                 |   |    |                         |   |    |
#  host data--------->o    |          host data--------->o    |
# (batch_size)    |        |         (batch_size)    |        |
#                 |--------|                         |--------|
#                 |        |                         |        |
#                 |    .   |                         |    .   |
#                 |    .   |                         |    .   |
#                 |    .   |                         |    .   |
#                 |        |                         |        |
#                 +--------+                         +--------+
#
def get_mux(layout, name, batch_size, width, height):
  mux = layout.create_code_region('./mux.csl', name, width, height)
  mux.set_param_all('size', batch_size)

  in_color = mux.color('in_color')
  out_color = mux.color('out_color')
  mux.set_param_all(in_color)
  mux.set_param_all(out_color)

  core_out_route = RoutingPosition().set_input([Route.RAMP]).set_output([Route.NORTH])
  forward_route = RoutingPosition().set_input([Route.SOUTH]).set_output([Route.NORTH])
  mux.paint_all(out_color, [core_out_route, forward_route])

  input_routes = RoutingPosition().set_output([Route.RAMP])
  output_routes = RoutingPosition().set_input([Route.RAMP])
  forward_port_routes = RoutingPosition().set_input([Route.SOUTH])

  size = batch_size * height
  in_port = mux.create_input_port(in_color, Edge.LEFT, [input_routes], size)
  out_port = mux.create_output_port(
      out_color, Edge.TOP, [output_routes, forward_port_routes], size
  )
  return (in_port, out_port, mux)
