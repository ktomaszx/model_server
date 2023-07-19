//*****************************************************************************
// Copyright 2022 Intel Corporation
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
//*****************************************************************************
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <thread>
#include <cstring>
#include <map>
#include <chrono>

#include "../../custom_node_interface.h"

#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace py::literals;

static constexpr const char* INPUT_TENSOR_NAME = "input";
static constexpr const char* OUTPUT_TENSOR_NAME = "output";

const int num_elements = 10 * 3 * 1024 * 1024;

class OutputWrapper {

    public:
    py::object model_instance;
    py::array output;
    OutputWrapper() {
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")(".\"");
        py::module_ script = py::module_::import("script");
        py::object OvmsPythonModel = script.attr("OvmsPythonModel");
        model_instance = OvmsPythonModel();
        model_instance.attr("initialize")();
    };

    uint8_t * getData() {return  reinterpret_cast<uint8_t*>(output.mutable_data());};
};

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    py::gil_scoped_acquire acquire;
    OutputWrapper * outputWrapper = new OutputWrapper();
    *customNodeLibraryInternalManager = (void*)outputWrapper;
    /*
    std::cout << "outputWrapper: " << outputWrapper << std::endl;
    std::cout << "customNodeLibraryInternalManager: " << *customNodeLibraryInternalManager << std::endl;
    */
    py::gil_scoped_release release;
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    py::gil_scoped_acquire acquire;
    std::cout << "Deinitizalizing data under address: " << customNodeLibraryInternalManager << std::endl;
    delete(customNodeLibraryInternalManager);
    py::gil_scoped_release release;
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {

    std::thread::id this_id = std::this_thread::get_id();
    // Inputs reading

    const CustomNodeTensor* inputTensor = nullptr;
    inputTensor = &(inputs[0]);
    int *input_c_arr = reinterpret_cast<int*>(inputTensor->data);
    uint64_t outputByteSize = sizeof(int) *  num_elements;
    int* output_buffer = (int*)malloc(outputByteSize);
    uint8_t* output_c_arr;
    OutputWrapper * outputWrapper = (OutputWrapper*)customNodeLibraryInternalManager;
    /*
    std::cout << "inputs: " << inputs << std::endl;
    std::cout << "customNodeLibraryInternalManager: " << customNodeLibraryInternalManager << std::endl;
    std::cout << "outputWrapper: " << outputWrapper << std::endl;
    */

    bool with_copy = false;
    py::gil_scoped_acquire acquire;
    try {
        py::str dummyDataOwner;
        py::array input_ndarray;
        auto start = std::chrono::high_resolution_clock::now();
        if (with_copy)
            input_ndarray = py::array(num_elements, input_c_arr);
        else
            input_ndarray = py::array(num_elements, input_c_arr, dummyDataOwner);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Ndarray creation time: " << duration.count() << " ms" << std::endl;

        py::dict input_dict("input"_a=input_ndarray);
        start = std::chrono::high_resolution_clock::now();
        py::dict output_dict = outputWrapper->model_instance.attr("execute")(input_dict);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Python execution time: " << duration.count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        if (with_copy) {
            output_c_arr = (uint8_t*)malloc(outputByteSize);
            py::array output_ndarray = output_dict["output"];
            const uint8_t* ndarray_buffer = reinterpret_cast<const uint8_t*>(output_ndarray.mutable_data());
            std::memcpy(output_c_arr, ndarray_buffer, outputByteSize);
        }
        else {
            outputWrapper->output = output_dict["output"];
            output_c_arr = outputWrapper->getData();
        }
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Ndarray extraction time: " << duration.count() << " ms" << std::endl;
    }
    catch (const py::error_already_set& error) {
        std::cout << "Error there mate: " << error.what() << std::endl;
        return 1;
    }
    py::gil_scoped_release release;

    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    if ((*outputs) == nullptr) {
        std::cout << "malloc has failed" << std::endl;
        free(output_buffer);
        return 1;
    }

    CustomNodeTensor& output = (*outputs)[0];
    output.name = "output";
    output.data = output_c_arr;
    //output.data = reinterpret_cast<uint8_t*>(output_buffer);
    output.dataBytes = outputByteSize;
    output.dimsCount = 1;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    output.dims[0] = num_elements;
    output.precision = I32;
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)[0].name = "input";
    (*info)[0].dimsCount = 1;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    (*info)[0].dims[0] = num_elements;
    (*info)[0].precision = I32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)[0].name = "output";
    (*info)[0].dimsCount = 1;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)[0].dims[0] = num_elements;
    (*info)[0].precision = I32;

    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    // TO DO: For zero copy data pointer in custom node tensor is owned by customNodeLibraryInternalManager and has not been created with malloc in execute()
    // need to work on that so we don't leak here
    //std::cout << "Freeing data under address: " << ptr << std::endl;
    //free(ptr);
    return 0;
}
