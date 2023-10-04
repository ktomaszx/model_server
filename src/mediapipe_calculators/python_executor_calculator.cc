//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <unordered_map>
#include "../ovms_py_tensor.hpp"
#include "../python_node_resources.hpp"

#include "mediapipe/framework/calculator_framework.h"

#include <pybind11/embed.h> // everything needed for embedding

namespace py = pybind11;
using namespace py::literals;

namespace mediapipe {

typedef std::vector<OvmsPyTensor> PyTensors;
typedef const std::unordered_map<const std::string, const PythonNodeResources>* PythonNodesResources;

const std::string INPUT_SIDE_PACKET_TAG = "PythonNodesResources";

class 

class PythonExecutorCalculator : public CalculatorBase {
PythonNodeResources nodeResources;
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(ERROR) << "PythonBackendCalculator::GetContract";
        cc->Inputs().Index(0).Set<PyTensors>();
        cc->Outputs().Index(0).Set<PyTensors>();
        cc->InputSidePackets().Tag(INPUT_SIDE_PACKET_TAG).Set<PythonNodesResources>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonBackendCalculator::Close";
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonBackendCalculator::Open";
        LOG(ERROR) << "Python node name:" << cc->NodeName();
        PythonNodesResources nodesResources = cc->InputSidePackets().Tag(INPUT_SIDE_PACKET_TAG);
        auto it = nodesResources->find(cc->NodeName());
        if (it == nodesResources.end())
            return absl::NotFoundError();
        nodeResources =  it->second;
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonBackendCalculator::Process";

        py::gil_scoped_acquire acquire;
        py::print("PYTHON: Acquired GIL");
        try {
            //pyobjectClass.attr("execute")();
            py::iterator it = nodeResources.userClass.attr("execute")(inputs);
            while (it != py::iterator::sentinel()) {
                result = cast<PYOBJECT>(*it);
                cc->Outputs().Index(0).Add(result, myTimestamp);
                ++it;
            }

            ov::Tensor in_tensor = cc->Inputs().Index(0).Get<ov::Tensor>();

            auto* out_tensor = new ov::Tensor(in_tensor.get_element_type(), in_tensor.get_shape());
            std::memcpy(out_tensor->data(), in_tensor.data(), in_tensor.get_byte_size());

            for (int i = 0; i < 10; i++) {
                float* pV = ((float*)out_tensor->data()) + i;
                *pV += 1.0f;
            }


            py::print("PYTHON: Released GIL");
        }
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(PythonExecutorCalculator);
}  // namespace mediapipe