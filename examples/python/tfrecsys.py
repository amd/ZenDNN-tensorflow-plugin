# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#
# ******************************************************************************

import os
import time
import wget
import numpy as np
import tensorflow as tf
import collections
import argparse
import math

def download_model():
  url = "https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_fp32_pretrained_model.pb"
  if not os.path.exists("wide_deep_fp32_pretrained_model.pb"):
    print("Downloading pre-trained model...")
    wget.download(url)
    print("\nDownload complete!")
  else:
    print("Model file already exists, skipping download.")

def input_fn(datafile, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  def _parse_function(proto):
    numeric_feature_names = ["numeric_1"]
    string_feature_names = ["string_1"]
    full_features_names = numeric_feature_names + string_feature_names
    feature_datatypes = [
     tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,
                    allow_missing=True)
    ] + [
     tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,
                    allow_missing=True)
    ]
    f = collections.OrderedDict(
      zip(full_features_names, feature_datatypes))
    parsed_features = tf.io.parse_example(proto, f)
    parsed_feature_vals_num = [tf.reshape(
      parsed_features["numeric_1"], shape=[-1, 13])]
    parsed_feature_vals_str = [
        tf.reshape(
            parsed_features["string_1"],
            shape=[-1, 2]
        ) for i in string_feature_names
    ]
    parsed_feature_vals = parsed_feature_vals_num + parsed_feature_vals_str
    return parsed_feature_vals

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TFRecordDataset([datafile])
  if shuffle:
    dataset = dataset.shuffle(buffer_size=20000)
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(_parse_function, num_parallel_calls=28)
  dataset = dataset.prefetch(batch_size*10)
  return dataset

def load_and_run_model(batch_size=512, warm_iter=10, num_batches=50):
  parser = argparse.ArgumentParser()
  parser.add_argument('--datafile', help='Path to the TFRecord data file')
  args = parser.parse_args()
  # Load the frozen graph.
  with tf.io.gfile.GFile("wide_deep_fp32_pretrained_model.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

  # Create a new graph and import the model into it.
  graph = tf.Graph()
  with graph.as_default():
    tf.import_graph_def(graph_def, name='import')

  # Retrieve the two input placeholders and the output tensor.
  with graph.as_default():
    placeholder_list = ['import/new_numeric_placeholder:0',
    'import/new_categorical_placeholder:0']
    input_tensor = [graph.get_tensor_by_name(name) for name in placeholder_list]
    output_name = "import/head/predictions/probabilities"
    output_tensor = graph.get_tensor_by_name("import/" + output_name + ":0" )

  # Generate synthetic input data:
  # Numeric data: shape [batch_size, 13] as float32.
  # Categorical data: shape [batch_size, 2] as int64.
  numeric_data = np.random.rand(batch_size, 13).astype(np.float32)
  categorical_data = np.random.randint(0, 10,
          size=(batch_size, 2)).astype(np.int64)

  total_infer_time = 0.0
  features_list = []
  config = tf.compat.v1.ConfigProto(log_device_placement=False)

  with tf.compat.v1.Session(config=config, graph=graph) as sess:
    res_dataset = input_fn(args.datafile, 1, False, batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(res_dataset)
    next_element = iterator.get_next()
    no_of_test_samples = sum(1
      for _ in tf.compat.v1.python_io.tf_record_iterator(
        args.datafile))
    no_of_batches = math.ceil(float(no_of_test_samples)/batch_size)
    for i in range(int(no_of_batches)):
      batch=sess.run(next_element)
      features=batch[0:3]
      features_list.append(features)

  with tf.compat.v1.Session(graph=graph) as sess:
    for i in range(num_batches):
      if i >= warm_iter:
        start_time = time.time()

      # Run inference with both inputs fed in the feed_dict.
      output = sess.run(output_tensor, dict(zip(input_tensor,
      [features_list[i][0], features_list[i][1]])))

  print('--------------------------------------------------')
  print("Output tensor shape:", output.shape)
  print("Output tensor value:", output)
  print('--------------------------------------------------')

if __name__ == "__main__":
  download_model()
  load_and_run_model()
