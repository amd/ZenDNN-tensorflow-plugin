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

from PIL import Image
from transformers import AutoImageProcessor, TFResNetForImageClassification
import tensorflow as tf
import urllib.request

# Load image.
urllib.request.urlretrieve("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n03954731_plane.JPEG", "airplane.jpeg")
image = Image.open("airplane.jpeg")

# Load processor and model.
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = TFResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Inputs.
inputs = image_processor(image, return_tensors="tf")

@tf.function
def predict():
  return model(**inputs)

# Inference.
logits = predict().logits
predicted_label = int(tf.math.argmax(logits, axis=-1))
print(model.config.id2label[predicted_label])
