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

from cerebras.geometry.geometry import IntRectangle, IntVector
from cerebras.sdk.runtime.sdkruntimepybind import (
    Color,
    Edge,
    Route,
    RoutingPosition,
    get_edge_routing,
)

def get_gemv(layout, name, width, height, batch_width, batch_height, control_ep, A):
  gemv = layout.create_code_region('./gemv.csl', name, width, height)

  ###########
  ### Colors
  ###########
  x_in = Color('x_in')
  b_in = Color('b_in')
  y_out = Color('y_out')

  ################
  ### Parameters
  ################
  gemv.set_param_all('width', batch_width)
  gemv.set_param_all('height', batch_height)
  gemv.set_param_all('control_ep', control_ep)
  gemv.set_param_all(x_in)

  ###########################
  ### Routing for vector 'x'
  ###########################
  # The bottom row's routing is slightly different because we no longer
  # need to forward data any further to the south. If we attempt to do that
  # backpressure will stall the execution as there is no one to consume the
  # stream on the south of the bottom row.
  x_routes_core = (
      RoutingPosition().set_input([Route.NORTH]).set_output([Route.RAMP, Route.SOUTH])
  )
  x_routes_bottom = (
      RoutingPosition().set_input([Route.NORTH]).set_output([Route.RAMP])
  )
  x_bottom_routes = get_edge_routing(Edge.BOTTOM, [x_routes_bottom])
  gemv.paint_all(x_in, [x_routes_core], [x_bottom_routes])

  #######################
  ### Input port for 'x'
  #######################
  x_size = batch_width * width
  x_port_routes = RoutingPosition().set_output([Route.RAMP, Route.SOUTH])
  x_port = gemv.create_input_port(x_in, Edge.TOP, [x_port_routes], x_size)

  #######################
  ### Input port for 'b'
  #######################
  b_size = batch_height * height
  b_rx_routes = RoutingPosition().set_output([Route.RAMP])
  b_port = gemv.create_input_port(b_in, Edge.LEFT, [b_rx_routes], b_size)

  ###############################
  ### Checkerboard pattern setup
  ###############################
  # As the vector 'b' flows horizontally through the gemv code region
  # it gets received by each PE, combined with the 'x' reduction for
  # that tile, and it is then forwarded to the neighbouring PEs to the
  # EAST using a different color. This means that initially at the left-most
  # column color 'b_in' and color 'y_out' have their allocated values. However,
  # in the next column, their values are swapped. That is, color 'b_in'
  # has the value of color 'y_out' and vice versa.
  receive_routes = RoutingPosition().set_input([Route.WEST]).set_output([Route.RAMP])
  sender_routes = RoutingPosition().set_input([Route.RAMP]).set_output([Route.EAST])

  # Since we have already specified the input port for 'b' above,
  # we also setup routing for the partial result 'y' along the left-most
  # column (i.e., the column that represents the input port for 'b').
  ul = IntVector(0, 0)
  lr = IntVector(1, height)
  hot_pes = IntRectangle(ul, lr)
  gemv.paint_range(hot_pes, y_out, [sender_routes])
  gemv.set_param_range(hot_pes, b_in)
  gemv.set_param_range(hot_pes, y_out)

  # We now alternate the routing and code parameters between colors
  # 'b_in' and 'y_out' to create the checkerboard pattern described above.
  for i in range(1, width):
    b_in, y_out = y_out, b_in
    ul = IntVector(i, 0)
    lr = IntVector(i + 1, height)
    hot_pes = IntRectangle(ul, lr)
    gemv.paint_range(hot_pes, b_in, [receive_routes])
    gemv.paint_range(hot_pes, y_out, [sender_routes])
    gemv.set_param_range(hot_pes, 'b_in', b_in)
    gemv.set_param_range(hot_pes, 'y_out', y_out)

  ######################################
  ### Output port for result vector 'y'
  ######################################
  y_tx_routes = RoutingPosition().set_input([Route.RAMP])
  y_port = gemv.create_output_port(y_out, Edge.RIGHT, [y_tx_routes], b_size, '_out')

  #########################
  ### Set the value of 'A'
  #########################
  # Finally, we set the value of 2D matrix 'A' across the code region's PEs.
  gemv.set_symbol_all('A', A, x_size, b_size)

  return (x_port, b_port, y_port, gemv)
