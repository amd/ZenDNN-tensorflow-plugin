//*******************************************************************************
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//*******************************************************************************
package org.tensorflow.benchmark;

import static org.tensorflow.internal.c_api.global.tensorflow.TF_LoadPluggableDeviceLibrary;

import java.io.IOException;
import java.lang.System;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;
import org.tensorflow.Graph;
import org.tensorflow.GraphOperation;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.Result;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.internal.c_api.TF_Library;
import org.tensorflow.internal.c_api.TF_Status;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.LongNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Ops;
import org.tensorflow.op.io.ReadFile;
import org.tensorflow.op.io.WriteFile;
import org.tensorflow.proto.GraphDef;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

public class RunWideDeeplarge {
  public static void main(String[] params) {
    // The libamdcpu_plugin_cc.so file should be present in LD_LIBRARY_PATH.
    String zentf_path = "libamdcpu_plugin_cc.so";

    // Get path to model.
    String modelPath = params[0];

    // Set input batch size.
    int batchSize = Integer.parseInt(params[1]);

    // Load saved model.
    byte[] graphDef = readAllBytesOrExit(Paths.get(modelPath));

    load_zentf(zentf_path);
    System.out.println("Using zendnn.");

    executeInceptionGraph(graphDef, batchSize);
  }

  private static void load_zentf(String filename) {
    TF_Status status = TF_Status.newStatus();
    TF_Library h = TF_LoadPluggableDeviceLibrary(filename, status);
    status.throwExceptionIfNotOK();
  }

  private static void executeInceptionGraph(
      byte[] graphDef, int batchSize) {
    try (Graph g = new Graph()) {
      GraphDef def = null;
      try {
        def = GraphDef.parseFrom(graphDef);
      } catch (IOException e) {
        System.out.println("Error: Could not parse graphDef");
      }
      g.importGraphDef(def);
      // The model requires 26 Categorical and 13 Numerical shape input.
      long[] numShape = {1L * batchSize, 13L};
      Shape tensorNumShape = Shape.of(numShape);
      long[] catShape = {26L * batchSize, 2L};
      Shape tensorCatShape = Shape.of(catShape);

      try (Session s = new Session(g)) {

        FloatNdArray numeric = NdArrays.ofFloats(tensorNumShape);
        LongNdArray category = NdArrays.ofLongs(tensorCatShape);

        for (long i = 0; i < batchSize; i++) {
          Random random = new Random();
          for (int j = 0; j < 13; j++) {
            numeric.setFloat(random.nextFloat(), i, j);
          }
        }

        for (long i = 0; i < 26 * batchSize; i++) {
          category.setLong(i % 26, i, 0);
          Random random = new Random();
          category.setLong(random.nextInt(1000), i, 1);
        }

        TFloat32 inputNumeric = TFloat32.tensorOf(numeric);
        TInt64 inputCategory = TInt64.tensorOf(category);

        Tensor resultTensor = s.runner()
                                  .feed("new_categorical_placeholder", inputCategory)
                                  .feed("new_numeric_placeholder", inputNumeric)
                                  .fetch("import/head/predictions/probabilities")
                                  .run()
                                  .get(0);

        System.out.println("Batch Size = " + batchSize);

        System.out.println("Output: ");

          float prob0 = resultTensor.asRawTensor().data().asFloats().getFloat(0);
          float prob1 = resultTensor.asRawTensor().data().asFloats().getFloat(1);
          int predicted = (prob0 > prob1) ? 0 : 1;
          System.out.println(predicted + " ");

        System.out.println("End of execution.");
      }
    }
  }

  private static byte[] readAllBytesOrExit(Path path) {
    try {
      return Files.readAllBytes(path);
    } catch (IOException e) {
      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
      System.exit(1);
    }
    return null;
  }
}