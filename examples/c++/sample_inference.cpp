/*******************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *******************************************************************************/

// This is the sample file to inference the Public CNN models.

// Required standard header files.
#include <sys/ipc.h>
#include <sys/shm.h>

#include <cstring>
#include <ctime>
#include <iostream>
#include <string>

// Required headers from TF.
#include <tensorflow/core/framework/graph.pb.h>         // for GraphDef.
#include <tensorflow/core/framework/tensor.h>           // for Tensor.
#include <tensorflow/core/framework/tensor_shape.h>     // for TensorShape.
#include <tensorflow/core/framework/tensor_shape.pb.h>  // for tensorflow.
#include <tensorflow/core/framework/tensor_types.h>     // for TTypes<>::Flat.
#include <tensorflow/core/framework/types.pb.h>         // for DT_FLOAT.
#include <tensorflow/core/platform/env.h>               // for ReadBinaryProto.
#include <tensorflow/core/platform/status.h>            // for Status.
#include <tensorflow/core/protobuf/config.pb.h>         // for ConfigProto.
#include <tensorflow/core/public/session.h>             // for NewSession.
#include <tensorflow/core/public/session_options.h>     // for SessionOptions.

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"

namespace tf = ::tensorflow;

int main(int argc, char const* argv[]) {
  // Load the plugin library.
  TF_Status* status = TF_NewStatus();
  TF_Library* tfplugin_lib =
      TF_LoadPluggableDeviceLibrary("libamdcpu_plugin_cc.so", status);

  if (argc < 4) {
    std::cout << "Please execute the run as - $ <object file name> <.pb model "
                 "file path> <input node> <output node>"
              << std::endl;
    return 0;
  }

  // Initialize required variables.
  std::string model_path = argv[1];
  std::string input_node = argv[2];
  std::string output_node = argv[3];
  int batch_size = argc > 4 ? atoi(argv[4]) : 640;
  int input_height = argc > 5 ? atoi(argv[5]) : 224;
  int input_width = argc > 6 ? atoi(argv[6]) : 224;
  int input_channels = argc > 7 ? atoi(argv[7]) : 3;
  int output_classes = argc > 8 ? atoi(argv[8]) : 1000;
  int inter_op_threads = 1;
  int intra_op_threads = 64;

  // Initialize TF session and load the model.
  tf::Session* session;
  tf::GraphDef graph_def;

  // Set any options if required.
  tf::SessionOptions options;
  tf::ConfigProto& config = options.config;
  config.set_use_per_session_threads(false);
  config.set_intra_op_parallelism_threads(intra_op_threads);
  config.set_inter_op_parallelism_threads(inter_op_threads);

  // Create a new session.
  TF_CHECK_OK(tf::NewSession(options, &(session)));
  // Load the model as a graph def.
  TF_CHECK_OK(tf::ReadBinaryProto(tf::Env::Default(), model_path, &graph_def));
  // Load the graph to the session.
  TF_CHECK_OK(session->Create(graph_def));

  // Initialize a vector with random values.
  int input_size = batch_size * input_height * input_width * input_channels;
  auto random_func = []() { return ((float)std::rand()) / ((float)RAND_MAX); };
  std::vector<float> random_vec(input_size);
  generate(begin(random_vec), end(random_vec), random_func);

  // Initialize a tensor and copy the data into it.
  tf::Tensor input_tensor(
      tf::DT_FLOAT,
      tf::TensorShape({batch_size, input_height, input_width, input_channels}));
  std::copy(begin(random_vec), end(random_vec),
            input_tensor.flat<float>().data());
  std::cout << "\nRandom input data: " << input_tensor.DebugString()
            << std::endl;

  // Initialize an input pair with input node name and input tensor.
  std::vector<std::pair<std::string, tf::Tensor>> input_pair = {
      {input_node, input_tensor}};
  // Initialize an output tensor to hold the output.
  std::vector<tensorflow::Tensor> output_tensor;

  // Run the benchmark on the model.
  int warmup = 10;
  int iter = 100;

  // Warmup steps (time not considered).
  for (int i = 0; i < warmup; i++) {
    session->Run(input_pair, {output_node}, {}, &output_tensor);
  }

  // Benchmarking.
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iter; i++) {
    session->Run(input_pair, {output_node}, {}, &output_tensor);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  float time_tmp = duration.count() / iter;
  std::time_t start_time = std::chrono::system_clock::to_time_t(start);
  std::time_t stop_time = std::chrono::system_clock::to_time_t(stop);
  std::cout << "Start  " << std::ctime(&start_time) << "Stop "
            << std::ctime(&stop_time) << "\n";

  std::cout << "\nExample Output data: " << output_tensor[0].DebugString()
            << std::endl;

  // If data requires to be copied back to a vector.
  int output_size = batch_size * output_classes;
  std::vector<float> output_vec(output_size);
  std::copy(output_tensor[0].flat<float>().data(),
            output_tensor[0].flat<float>().data() + output_size,
            output_vec.data());

  // Print benchmarking result.
  std::cout << "\nTime taken: " + std::to_string(time_tmp) << std::endl;
  std::cout << "FPS for " + std::to_string(batch_size) +
                   " images: " + std::to_string(batch_size / time_tmp * 1000)
            << std::endl;

  return 0;
}
