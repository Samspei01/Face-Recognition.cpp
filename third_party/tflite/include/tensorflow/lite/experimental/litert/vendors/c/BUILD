# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "litert_compiler_plugin",
    hdrs = ["litert_compiler_plugin.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
    ],
)

cc_library(
    name = "litert_compiler_plugin_api",
    hdrs = ["litert_compiler_plugin_api.h"],
    deps = [
        ":litert_compiler_plugin",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
    ],
)

cc_library(
    name = "litert_dispatch_c_api",
    hdrs = [
        "litert_dispatch.h",
        "litert_dispatch_api.h",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/runtime/dispatch",
    ],
)

exports_files(srcs = glob(["litert_*.h"]))
