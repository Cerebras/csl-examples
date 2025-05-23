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
import json
import subprocess
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

matplotlib.use('Agg')


def game_of_life_ref(initial_state, num_generations):
  """Compute reference to check WSE result for game of life generation"""

  x_dim = initial_state.shape[0]
  y_dim = initial_state.shape[1]

  ref_states = np.zeros((x_dim, y_dim, num_generations))
  ref_states[:,:,0] = initial_state

  for gen in range(1, num_generations):
    for i in range(x_dim):
      for j in range(y_dim):
        total = (0 if (i == 0)                           else ref_states[i-1,j,  gen-1]) \
              + (0 if (i == x_dim-1)                     else ref_states[i+1,j,  gen-1]) \
              + (0 if (j == 0)                           else ref_states[i,  j-1,gen-1]) \
              + (0 if (j == y_dim-1)                     else ref_states[i,  j+1,gen-1]) \
              + (0 if ((i == 0)     or (j == 0))         else ref_states[i-1,j-1,gen-1]) \
              + (0 if ((i == 0)     or (j == y_dim-1))   else ref_states[i-1,j+1,gen-1]) \
              + (0 if ((i == x_dim-1) or (j == 0))       else ref_states[i+1,j-1,gen-1]) \
              + (0 if ((i == x_dim-1) or (j == y_dim-1)) else ref_states[i+1,j+1,gen-1])

        if (ref_states[i, j, gen-1] == 1):
          ref_states[i, j, gen] = 1 if (total in (2, 3)) else 0
        else:
          ref_states[i, j, gen] = 1 if (total == 3) else 0

  return ref_states


def show_ascii_animation(states):
  """Generate a command-line ASCII animation"""

  num_generations = states.shape[2]
  try:
    for i in range(num_generations):
      subprocess.run(['clear'], shell=True, check=True)
      print(f'Generation {i}:\n')
      for row in states[:, :, i]:
        print(' '.join(['#' if cell else '.' for cell in row]))
      print('\nPress Ctrl+C to exit.')
      time.sleep(0.1)  # Wait for 0.1 seconds before displaying the next frame
  except KeyboardInterrupt:
    print('\nAnimation stopped.')


def save_animation(states, fname):
  """Save an animation as a GIF"""

  fig, ax = plt.subplots()
  ax.set_xticks([])
  ax.set_yticks([])
  ax.axis('off')

  frame_image = ax.imshow(states[:, :, 0], cmap='Greys', vmin=0, vmax=1)

  def update_plot(frame_index):
    frame_image.set_data(states[:, :, frame_index])
    return [frame_image]

  anim = FuncAnimation(
    fig,
    update_plot,
    frames=states.shape[2],
    interval=100,  # 0.1 seconds per frame
    blit=True
  )

  output_file = fname + '.gif'
  anim.save(output_file, writer=PillowWriter(fps=10))


def create_initial_state(state_type, x_dim, y_dim):
  """Generate intitial state for Game of Life"""

  initial_state = np.zeros((x_dim, y_dim), dtype=np.uint32)

  if state_type == 'glider':
    assert x_dim >= 4 and y_dim >=4, \
           'For glider initial state, x_dim and y_dim must be at least 4'

    glider = np.array([[0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1]])

    for i in range(x_dim//4):
      for j in range(y_dim//4):
        if i%2 == 0 and j%2 == 0:
          initial_state[4*i:4*i+3, 4*j:4*j+3] = glider
        elif i%2 == 0 and j%2 == 1:
          initial_state[4*i:4*i+3, 4*j:4*j+3] = glider[:,::-1]
        elif i%2 == 1 and j%2 == 0:
          initial_state[4*i:4*i+3, 4*j:4*j+3] = glider[::-1,:]
        elif i%2 == 1 and j%2 == 1:
          initial_state[4*i:4*i+3, 4*j:4*j+3] = glider[::-1,:]

  else: # state_type == 'random'
    np.random.seed(seed=7)
    initial_state = np.random.binomial(1, 0.5, (x_dim, y_dim)).astype(np.uint32)

  return initial_state


def main():
  """Main method to run the example code."""

  # Read arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', help='the test compile output dir', required=True)
  parser.add_argument('--cmaddr', help='IP:port for CS system')
  parser.add_argument('--iters', type=int, default=10, help='Number of generations (default: 10)')
  parser.add_argument('--initial-state', choices=['glider', 'random'], default='glider',
    help='Specify the initial state of the system (default: glider)'
  )
  parser.add_argument('--save-animation', action='store_true',
    help="Save animated GIF of states"
  )
  parser.add_argument('--show-ascii-animation', action='store_true',
    help="Show ascii animation of states"
  )
  args = parser.parse_args()

  # Get matrix dimensions from compile metadata
  with open(f'{args.name}/out.json', encoding='utf-8') as json_file:
    compile_data = json.load(json_file)

  # PE grid dimensions
  x_dim = int(compile_data['params']['x_dim'])
  y_dim = int(compile_data['params']['y_dim'])

  # Number of generations
  iters = args.iters

  initial_state = create_initial_state(args.initial_state, x_dim, y_dim)

  # Construct a runner using SdkRuntime
  runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

  states_symbol = runner.get_id('states')

  # Load and run the program
  runner.load()
  runner.run()

  print('Copy initial state to device...')
  # Copy initial state into all PEs
  runner.memcpy_h2d(states_symbol, initial_state.flatten(), 0, 0, x_dim, y_dim, 1,
    streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
    nonblock=False)

  print(f'Run for {iters} generations...')
  # Launch the generate function on device
  runner.launch('generate', np.uint16(iters), nonblock=False)

  # Copy states back
  states_result = np.zeros([x_dim * y_dim * iters], dtype=np.uint32)
  runner.memcpy_d2h(states_result, states_symbol, 0, 0, x_dim, y_dim, iters, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

  # Stop the program
  runner.stop()

  print('Create output...')

  # Reshape states results to x_dim x y_dim frames
  all_states = states_result.reshape((x_dim, y_dim, iters))

  # Loop through the frames and display them
  if args.show_ascii_animation:
    show_ascii_animation(all_states)

  # Generate animated GIF of generations
  if args.save_animation:
    save_animation(all_states, 'game_of_life')

  print('Create reference solution...')
  ref_states = game_of_life_ref(initial_state, iters)

  # Test that wafer output is equal to the reference
  np.testing.assert_equal(ref_states, all_states)
  print('SUCCESS!')

if __name__ == '__main__':
  main()
