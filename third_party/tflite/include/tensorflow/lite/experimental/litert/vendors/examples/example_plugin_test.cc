// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"

namespace {

UniqueLiteRtCompilerPlugin GetDummyPlugin() {
  LiteRtCompilerPlugin dummy_plugin;
  LITERT_CHECK_STATUS_OK(LiteRtCreateCompilerPlugin(&dummy_plugin));
  ABSL_CHECK_NE(dummy_plugin, nullptr);
  return UniqueLiteRtCompilerPlugin(dummy_plugin);
}

TEST(TestDummyPlugin, GetConfigInfo) {
  ASSERT_STREQ(LiteRtGetCompilerPluginSocManufacturer(),
               "ExampleSocManufacturer");

  auto plugin = GetDummyPlugin();

  LiteRtParamIndex num_supported_soc_models;
  LITERT_ASSERT_STATUS_OK(LiteRtGetNumCompilerPluginSupportedSocModels(
      plugin.get(), &num_supported_soc_models));
  ASSERT_EQ(num_supported_soc_models, 1);

  const char* soc_model_name;
  LITERT_ASSERT_STATUS_OK(LiteRtGetCompilerPluginSupportedSocModel(
      plugin.get(), 0, &soc_model_name));
  ASSERT_STREQ(soc_model_name, "ExampleSocModel");
}

TEST(TestCallDummyPlugin, PartitionSimpleMultiAdd) {
  auto plugin = GetDummyPlugin();
  auto model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_STATUS_OK(LiteRtCompilerPluginPartitionModel(
      plugin.get(), model.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Vec();

  ASSERT_EQ(selected_ops.size(), 2);
  ASSERT_EQ(selected_ops[0]->op_code, kLiteRtOpCodeTflMul);
  ASSERT_EQ(selected_ops[1]->op_code, kLiteRtOpCodeTflMul);
}

TEST(TestCallDummyPlugin, CompileMulSubgraph) {
  auto plugin = GetDummyPlugin();
  auto model = litert::testing::LoadTestFileModel("mul_simple.tflite");

  LITERT_ASSERT_RESULT_OK_ASSIGN(auto subgraph,
                                 litert::internal::GetSubgraph(model.Get()));

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_STATUS_OK(LiteRtCompilerPluginCompile(
      plugin.get(), /*soc_model=*/nullptr, &subgraph, 1, &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetCompiledResultByteCode(compiled, &byte_code, &byte_code_size));

  std::string byte_code_string(reinterpret_cast<const char*>(byte_code),
                               byte_code_size);
  ASSERT_EQ(byte_code_string, "Partition_0_with_2_muls:");

  const void* op_data;
  size_t op_data_size;

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetCompiledResultCallInfo(compiled, 0, &op_data, &op_data_size));

  std::string op_data_string(reinterpret_cast<const char*>(op_data),
                             op_data_size);
  ASSERT_EQ(op_data_string, "Partition_0");

  LiteRtDestroyCompiledResult(compiled);
}

}  // namespace
