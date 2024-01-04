# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import Sequential
import numpy as np

np.random.seed(1988)
input_dim = 16
input_shape = (input_dim, input_dim, 3)
depth_multiplier = 1
kernel_height = 3
kernel_width = 3

# Define a model
model = Sequential()
model.add(
    DepthwiseConv2D(
        depth_multiplier=depth_multiplier,
        kernel_size=(kernel_height, kernel_width),
        input_shape=input_shape,
        padding="valid",
        strides=(1, 1),
        activation='relu',
        use_bias=True,
    )
)

# Set some random weights
model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

# Test the keras model
x = np.random.rand(1, input_dim, input_dim, 3)
model.summary()
model_out = model.predict(x)
print('Model output shape:', model_out.shape)
