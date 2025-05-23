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
    get_edge_routing,
)

#  The demux adaptor is a single-PE code region that is responsible for controlling
#  the behavior of the demux layer (see below).
#
#  Specifically, the demux adaptor forwards all the data from the host
#  to the demux layer while injecting control signals after every 'batch_size'
#  number of wavelets. These control signals help the demux layer distribute the
#  data evenly across a 1D vector of PEs such that each PE in the demux layer
#  forwards 'batch_size' wavelets to the user's port.
#
#                  Demux
#                 adaptor                     Demux
#                +-------+      +--------------------------------------+
#                |       |      |       |       |                      |
# host data----->|       |----->|-->o   |-->o   |       ...            |
#                |       |      |   |   |   |   |                      |
#                +-------+      +---|---+---|---+----------------------+
#                                   |       |
#                                   V       V
#                                  batch
#                                  size
#
#                  Demux
#                 adaptor                     Demux
#                +-------+      +--------------------------------------+
#                |       | ctrl |       |       |                      |
# host data----->|       |----->|-->o---|-->o   |       ...            |
#                |       |      |       |   |   |                      |
#                +-------+      +-------+---|---+----------------------+
#                                           |
#                                           V
#                                          batch
#                                          size
def get_demux_adaptor(layout, name, batch_size, num_batches):
  demux_adaptor = layout.create_code_region('./demux_adaptor.csl', name, 1, 1)
  demux_adaptor.set_param_all('batch_size', batch_size)
  demux_adaptor.set_param_all('num_batches', num_batches)

  in_color = demux_adaptor.color('in_color')
  out_color = demux_adaptor.color('out_color')
  demux_adaptor.set_param_all(in_color)
  demux_adaptor.set_param_all(out_color)

  input_routes = RoutingPosition().set_output([Route.RAMP])
  output_routes = RoutingPosition().set_input([Route.RAMP])

  size = batch_size * num_batches
  in_port = demux_adaptor.create_input_port(in_color, Edge.LEFT, [input_routes], size)
  out_port = demux_adaptor.create_output_port(
      out_color, Edge.RIGHT, [output_routes], size
  )
  return (in_port, out_port, demux_adaptor)



# The goal of the demux layer is to connect an input stream from the host
# to a port that spans more than a single PE. That's because I/O streams
# to/from the host go through a single PE device port (in the future, this
# restriction can be lifted). This means that if a user wants to stream
# data from the host to a multi-PE port (which is the case for the 'x'
# and 'b' vectors in this gemv tutorial) then data need to be demultiplexed
# from a single-PE stream to a multi-PE stream.
#
# The demux layer achieves that in combination with the previous demux
# adaptor layer, by forwarding the first 'batch_size' number of wavelets
# from the first PE, the next 'batch_size' number of wavelets from the
# second PE and so on (see diagram above).
#
# This is achieved by utilizing the switching capability of the WSE where
# a control wavelet sent by the demux adaptor, instructs the PE router
# to move to a new routing position (see diagram above).
#
# The x demux layer differs from the b demux layer in that it is positioned
# horizontally. There is no need to do that but it helps to demonstrate a
# variation of the layer with different routing charachteristics.
#
# In addition, the x demux layer will also enable a control entry point
# after 'batch_size' wavelets are sent. This entry point informs the
# gemv kernel that the reduction of the x vector is done for a given tile
# which means that the b vector can now be added to the result.
def get_x_demux(layout, name, batch_size, width, height, entry_point):
  demux = layout.create_code_region('./demux.csl', name, width, height)
  demux.set_param_all('size', batch_size)

  demux.set_param_all('has_sentinel', 1)
  demux.set_param_all('entry_point', entry_point)

  in_color = demux.color('in_color')
  out_color = demux.color('out_color')
  demux.set_param_all(in_color)
  demux.set_param_all(out_color)

  # All PEs begin at pos1. This means that only the left-most PE
  # (i.e., the PE associated with the demux layer's input port) is able
  # to forward data to the gemv kernel at the beginning.
  # Once a control wavelet lands, each PE that receives it moves to pos2
  # which will effectively forward all remaining wavelets to the rest
  # of the PEs in the demux layer.
  #
  # Finally, the right-most PE doesn't need pos2 because it forwards
  # the last batch of data (i.e., no more data need to be forwarded).
  pos1 = RoutingPosition().set_input([Route.WEST]).set_output([Route.RAMP])
  pos2 = RoutingPosition().set_input([Route.WEST]).set_output([Route.EAST])
  edge_route = get_edge_routing(Edge.RIGHT, [pos1])
  demux.paint_all(in_color, [pos1, pos2], [edge_route])

  input_routes = RoutingPosition().set_output([Route.RAMP])
  output_routes = RoutingPosition().set_input([Route.RAMP])

  size = batch_size * width * height
  blah = RoutingPosition().set_output([Route.EAST])
  in_port = demux.create_input_port(in_color, Edge.LEFT, [input_routes, blah], size)
  out_port = demux.create_output_port(out_color, Edge.BOTTOM, [output_routes], size)
  return (in_port, out_port, demux)



# Same as the x demux layer but with two key differences:
#
#  - It is positioned vertically and therefore routing is different.
#  In the future, the SdkLayout API will support a 'flip' operation
#  on code regions which will allow us to re-use the x demux layer
#  by simply flipping it.
#
#  - The b demux layer does not need to send a control signal to the
#  gemv code regions because no more action is needed once the 'b'
#  vector is done being streamed through the gemv code region.
def get_b_demux(layout, name, batch_size, width, height):
  demux = layout.create_code_region('./demux.csl', name, width, height)
  demux.set_param_all('size', batch_size)

  demux.set_param_all('has_sentinel', 0)
  demux.set_param_all('entry_point', 0)

  in_color = demux.color('in_color')
  out_color = demux.color('out_color')
  demux.set_param_all(in_color)
  demux.set_param_all(out_color)

  core_out_route = RoutingPosition().set_input([Route.NORTH]).set_output([Route.RAMP])
  forward_route = RoutingPosition().set_input([Route.NORTH]).set_output([Route.SOUTH])
  edge_route = get_edge_routing(Edge.BOTTOM, [core_out_route])
  demux.paint_all(in_color, [core_out_route, forward_route], [edge_route])

  input_routes = RoutingPosition().set_output([Route.RAMP])
  output_routes = RoutingPosition().set_input([Route.RAMP])

  size = batch_size * width * height
  blah = RoutingPosition().set_output([Route.SOUTH])
  in_port = demux.create_input_port(in_color, Edge.TOP, [input_routes, blah], size)
  out_port = demux.create_output_port(out_color, Edge.RIGHT, [output_routes], size)
  return (in_port, out_port, demux)
