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
    name = "model",
    srcs = [
        "model.cc",
        "//tensorflow/lite/experimental/litert/c:litert_model_srcs",
    ],
    hdrs = [
        "model.h",
        "//tensorflow/lite/experimental/litert/c:litert_model_hdrs",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "model_test",
    srcs = ["model_test.cc"],
    deps = [
        ":model",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_load",
    srcs = ["model_load.cc"],
    hdrs = ["model_load.h"],
    deps = [
        ":model",
        ":model_util",
        "//tensorflow/compiler/mlir/lite:allocation",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "model_file_test",
    srcs = ["model_file_test.cc"],
    data = ["//tensorflow/lite/experimental/litert/test:tflite_test_data"],
    deps = [
        ":model",
        ":model_load",
        ":model_serialize",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/core:graph_tools",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_serialize",
    srcs = ["model_serialize.cc"],
    hdrs = ["model_serialize.h"],
    deps = [
        ":model",
        ":model_util",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "model_util",
    srcs = ["model_util.cc"],
    hdrs = ["model_util.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
