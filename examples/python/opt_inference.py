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

from transformers import AutoTokenizer, TFOPTForCausalLM
import tensorflow as tf

# Load tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = TFOPTForCausalLM.from_pretrained("facebook/opt-350m")

# Inputs.
prompt = "Are you conscious? Can you talk?"
input_ids = tokenizer(prompt, return_tensors='tf').input_ids

@tf.function
def generate():
  return model.generate(input_ids, max_length=20)

# Inference.
outputs = generate()
decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True)
                   for output in outputs]

for result in decoded_outputs:
  print(result)
