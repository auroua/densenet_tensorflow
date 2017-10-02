import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

densnet_path = os.path.join(dirname, '..', 'dense_net')
builder_path = os.path.join(dirname, '..', 'builder')
inputs_path = os.path.join(dirname, '..', 'input_piplines')
protos_path = os.path.join(dirname, '..', 'protos')

add_path(densnet_path)
add_path(builder_path)
add_path(inputs_path)
add_path(protos_path)
