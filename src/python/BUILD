#
# Copyright (c) 2024 Intel Corporation
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
#

load("//src:common_settings.bzl", "COMMON_STATIC_LIBS_COPTS", "COMMON_STATIC_LIBS_LINKOPTS", "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")

pybind_extension(
    name = "pyovms",
    srcs = ["ovms_py_tensor_py.cpp", # TODO consider naming
            "ovms_py_tensor.hpp",
            "ovms_py_tensor.cpp"],
    visibility = ["//visibility:public"],
)

mediapipe_proto_library(
    name = "pythonexecutorcalculatoroptions_proto", # pythonexecutorcalculatoroptions_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["python_executor_calculator_options.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "libovmspythonmodule",
    hdrs = ["pythonnoderesources.hpp",
            "ovms_py_tensor.hpp",
            "python_backend.hpp",
            "pythoninterpretermodule.hpp",
            ],
    srcs = ["python_backend.cpp",
            "ovms_py_tensor.cpp",
            "pythonnoderesources.cpp",
            "utils.hpp",
            "pythoninterpretermodule.cpp",
            "python_executor_calculator.cc",
            ],
    deps = [
        "pythonexecutorcalculatoroptions_cc_proto",
        "@com_github_grpc_grpc//:grpc++", # TODO find <google/protobuf/port_def.inc>
        "@python3_linux//:python3-lib",
        "@pybind11//:pybind11_embed",
        "@mediapipe//mediapipe/framework:calculator_framework", # TODO only for calculator
        "//src:cpp_headers", # TODO make ovmslib depend on that
        "//src:libovmslogging",
        "//src:libovmsmediapipe_utils"
    ],
    copts = COMMON_STATIC_LIBS_COPTS + select({
            "//conditions:default": [],
            "//src:fuzzer_build" : COMMON_FUZZER_COPTS,
        }),
    visibility = [
        "//visibility:public",
    ],
    linkopts = COMMON_STATIC_LIBS_LINKOPTS + select({
        "//conditions:default": [],
        "//src:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
    }),
    alwayslink = 1,
    data = [":pyovms.so"],
)
