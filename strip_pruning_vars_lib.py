# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities to remove pruning-related ops and variables from a GraphDef.
"""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.platform import gfile
from google.protobuf import text_format
from tensorflow.python.framework import importer
from tensorflow.python import pywrap_tensorflow
import re

import mobilenet_v1_prune
import tensorflow as tf


def _node_name(tensor_name):
  """Remove the trailing ':0' from the variable name."""
  if ':' not in tensor_name:
    return tensor_name

  return tensor_name.split(':')[0]


def _tensor_name(node_name):
  """Appends the :0 in the op name to get the canonical tensor name."""
  if ':' in node_name:
    return node_name

  return node_name + ':0'


def _get_masked_weights(input_graph_def):
  """Extracts masked_weights from the graph as a dict of {var_name:ndarray}."""
  input_graph = ops.Graph()
  with input_graph.as_default():
    importer.import_graph_def(input_graph_def, name='')

    with session.Session(graph=input_graph) as sess:
      masked_weights_dict = {}
      for node in input_graph_def.node:
        if 'masked_weight' in node.name:
          masked_weight_val = sess.run(
              sess.graph.get_tensor_by_name(_tensor_name(node.name)))
          logging.info(
              '%s has %d values, %1.2f%% zeros \n', node.name,
              np.size(masked_weight_val),
              100 - float(100 * np.count_nonzero(masked_weight_val)) /
              np.size(masked_weight_val))
          masked_weights_dict.update({node.name: masked_weight_val})
  return masked_weights_dict


def strip_pruning_vars_fn(input_graph_def, output_node_names):
  """Removes mask variable from the graph.

  Replaces the masked_weight tensor with element-wise multiplication of mask
  and the corresponding weight variable.

  Args:
    input_graph_def: A GraphDef in which the variables have been converted to
      constants. This is typically the output of
      tf.graph_util.convert_variables_to_constant()
    output_node_names: List of name strings for the result nodes of the graph

  Returns:
    A GraphDef in which pruning-related variables have been removed
  """
  masked_weights_dict = _get_masked_weights(input_graph_def)
  pruned_graph_def = graph_pb2.GraphDef()

  # Replace masked_weight with a const op containing the
  # result of tf.multiply(mask,weight)
  for node in input_graph_def.node:
    output_node = node_def_pb2.NodeDef()
    if 'masked_weight' in node.name:
      output_node.op = 'Const'
      output_node.name = node.name
      dtype = node.attr['T']
      data = masked_weights_dict[node.name]
      output_node.attr['dtype'].CopyFrom(dtype)
      output_node.attr['value'].CopyFrom(
          attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(data)))

    else:
      output_node.CopyFrom(node)
    pruned_graph_def.node.extend([output_node])

  # Remove stranded nodes: mask and weights
  return graph_util.extract_sub_graph(pruned_graph_def, output_node_names)


def graph_def_from_checkpoint(checkpoint_dir, output_node_names):
  """Converts checkpoint data to GraphDef.

  Reads the latest checkpoint data and produces a GraphDef in which the
  variables have been converted to constants.

  Args:
    checkpoint_dir: Path to the checkpoints.
    output_node_names: List of name strings for the result nodes of the graph.

  Returns:
    A GraphDef from the latest checkpoint

  Raises:
    ValueError: if no checkpoint is found
  """
  checkpoint_path = saver_lib.latest_checkpoint(checkpoint_dir)
  if checkpoint_path is None:
    raise ValueError('Could not find a checkpoint at: {0}.'
                     .format(checkpoint_dir))

  saver_for_restore = saver_lib.import_meta_graph(
      checkpoint_path + '.meta', clear_devices=True)
  with session.Session() as sess:
    saver_for_restore.restore(sess, checkpoint_path)
    graph_def = ops.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph_def, output_node_names)

  return output_graph_def


def _parse_input_graph_proto(input_graph, input_binary):
  """Parser input tensorflow graph into GraphDef proto."""
  if not gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1
  input_graph_def = graph_pb2.GraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  return input_graph_def

def _has_no_variables(sess):
  """Determines if the graph has any variables.

  Args:
    sess: TensorFlow Session.

  Returns:
    Bool.
  """
  for op in sess.graph.get_operations():
    if op.type.startswith("Variable") or op.type.endswith("VariableOp"):
      return False
  return True
def graph_def_from_pb(input_graph, input_binary, input_checkpoint, output_node_names, output_graph=None):
  """Converts checkpoint data to GraphDef.

  Reads the latest checkpoint data and produces a GraphDef in which the
  variables have been converted to constants.

  Args:
    checkpoint_dir: Path to the checkpoints.
    output_node_names: List of name strings for the result nodes of the graph.

  Returns:
    A GraphDef from the latest checkpoint

  Raises:
    ValueError: if no checkpoint is found
  """
  input_graph_def = _parse_input_graph_proto(input_graph, input_binary)
  if input_graph_def:
      _ = importer.import_graph_def(input_graph_def, name="")

  with session.Session() as sess:
      var_list = {}
      reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
      var_to_shape_map = reader.get_variable_to_shape_map()

      # List of all partition variables. Because the condition is heuristic
      # based, the list could include false positives.
      all_parition_variable_names = [
          tensor.name.split(":")[0]
          for op in sess.graph.get_operations()
          for tensor in op.values()
          if re.search(r"/part_\d+/", tensor.name)
      ]
      has_partition_var = False

      for key in var_to_shape_map:
          try:
              tensor = sess.graph.get_tensor_by_name(key + ":0")
              if any(key in name for name in all_parition_variable_names):
                  has_partition_var = True
          except KeyError:
              # This tensor doesn't exist in the graph (for example it's
              # 'global_step' or a similar housekeeping element) so skip it.
              continue
          var_list[key] = tensor

      try:
          saver = saver_lib.Saver(
              var_list=var_list)
      except TypeError as e:
          # `var_list` is required to be a map of variable names to Variable
          # tensors. Partition variables are Identity tensors that cannot be
          # handled by Saver.
          if has_partition_var:
              print("Models containing partition variables cannot be converted "
                    "from checkpoint files. Please pass in a SavedModel using "
                    "the flag --input_saved_model_dir.")
              return -1
          # Models that have been frozen previously do not contain Variables.
          elif _has_no_variables(sess):
              print("No variables were found in this model. It is likely the model "
                    "was frozen previously. You cannot freeze a graph twice.")
              return 0
          else:
              raise e

      saver.restore(sess, input_checkpoint)

      output_graph_def = graph_util.convert_variables_to_constants(
          sess,
          input_graph_def,
          output_node_names.replace(" ", "").split(",")

      )

      # Write GraphDef to file if output path has been given.
      if output_graph:
          with gfile.GFile(output_graph, "wb") as f:
              f.write(output_graph_def.SerializeToString())
  return output_graph_def
