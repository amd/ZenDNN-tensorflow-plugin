# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""zentf: A TensorFlow extension for AMD EPYC CPUs."""

import sys

# Import build info generated at wheel build time
try:
  from ._build_info import __zentf_commit__, __zendnn_version__, __tf_version__
except ImportError:
  __zentf_commit__ = "unknown"
  __zendnn_version__ = "unknown"
  __tf_version__ = "unknown"

# Get package version from metadata
if sys.version_info >= (3, 8):
  from importlib import metadata
  try:
    __version__ = metadata.version('zentf')
  except metadata.PackageNotFoundError:
    __version__ = '5.2.0'
else:
  __version__ = '5.2.0'


def show_config():
  """Return a string describing the zentf build configuration.

  Returns:
    str: A formatted string containing version and build information.
  """
  # Determine if zendnn_version is a commit hash or version tag
  if __zendnn_version__ and len(__zendnn_version__) <= 12 and all(c in \
    '0123456789abcdef' for c in __zendnn_version__):
    zendnn_label = f"AMD ZenDNNL ( Git Hash {__zendnn_version__} )"
  else:
    zendnn_label = f"AMD ZenDNNL v{__zendnn_version__}"

  config = f"""zentf Version: {__version__}
zentf built with:
  - Commit-id: {__zentf_commit__}
  - TensorFlow: {__tf_version__}
Third_party libraries:
  - {zendnn_label}
"""
  return config


__config__ = show_config()
