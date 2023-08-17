#*******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#*******************************************************************************

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
import numpy as np

input_shape=(28,28,1)

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Flatten())
model.add(Dense(10, activation="relu",use_bias=True,bias_initializer='zeros'))
model.summary()

x = np.random.rand(1,28,28,1)
model_out = model.predict(x)
print('Model output shape:', model_out.shape)
