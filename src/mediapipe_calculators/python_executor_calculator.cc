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
#include "../python/ovms_py_tensor.hpp"
#include "../mediapipe_internal/pythonnoderesource.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop

#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace ovms;

namespace mediapipe {

typedef std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>> PythonNodesResources;

const std::string INPUT_SIDE_PACKET_TAG = "PYTHON_NODE_RESOURCES";

typedef std::unique_ptr<OvmsPyTensor> OvmsPyTensorPtr;

class PythonExecutorCalculator : public CalculatorBase {
std::shared_ptr<PythonNodeResource> nodeResources;
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(ERROR) << "PythonBackendCalculator::GetContract";
        //RET_CHECK(!cc->Inputs().GetTags().empty());
        //RET_CHECK(!cc->Outputs().GetTags().empty());
        for (auto& input : cc->Inputs()) {
            input.Set<OvmsPyTensorPtr>();
        }
        for (auto& output : cc->Outputs()) {
            output.Set<OvmsPyTensorPtr>();
        }
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
        PythonNodesResources nodesResourcesPtr = cc->InputSidePackets().Tag(INPUT_SIDE_PACKET_TAG).Get<PythonNodesResources>();
        auto it = nodesResourcesPtr.find(cc->NodeName());
        if (it == nodesResourcesPtr.end())
            return absl::NotFoundError("error lol");
        nodeResources =  it->second;
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        py::gil_scoped_acquire acquire;
        try {
            LOG(ERROR) << "PythonBackendCalculator::Process";
            py::print("PYTHON: Acquired GIL");
            std::vector<py::object> pyInputs;
            for (const std::string& tag : cc->Inputs().GetTags()) {
                LOG(ERROR) << tag;
                const OvmsPyTensorPtr& pyTensor = cc->Inputs().Tag(tag).Get<OvmsPyTensorPtr>();
                py::object pyInput = nodeResources->tensorClass->attr("create_from_data")(pyTensor->name, pyTensor->ptr, pyTensor->userShape, pyTensor->datatype, pyTensor->size);
                pyInputs.push_back(pyInput);
            }
           
            //py::list pyOutputs = nodeResources->nodeResourceObject->attr("execute")(pyInputs);
            py::object pyOutput = nodeResources->nodeResourceObject->attr("execute")(pyInputs);
            std::unique_ptr<OvmsPyTensorPtr> outputPtr = std::make_unique<OvmsPyTensorPtr>(
                std::make_unique<OvmsPyTensor>(pyOutput.attr("name").cast<std::string>(), py::cast<py::buffer>(pyOutput.attr("ref_obj")))
            );
            cc->Outputs().Tag("OUTPUT").Add(outputPtr.release(), cc->InputTimestamp());

/*
            for (const std::string& tag : cc->Outputs().GetTags()) {
                for (py::handle pyOutput : pyOutputs) {
                    LOG(ERROR) << tag << " == " << pyOutput.attr("name").cast<std::string>();
                    if (pyOutput.attr("name").cast<std::string>() == tag) {
                        std::unique_ptr<OvmsPyTensorPtr> outputPtr = std::make_unique<OvmsPyTensorPtr>(
                            std::make_unique<OvmsPyTensor>(pyOutput.attr("name").cast<std::string>(), py::cast<py::buffer>(pyOutput))
                        );
                        cc->Outputs().Tag(tag).Add(outputPtr.release(), cc->InputTimestamp());
                    }
                }
            }
            */
            py::print("PYTHON: Released GIL");
        } catch (const pybind11::error_already_set& e) {
            LOG(ERROR) << "PythonBackendCalculator::Process - Error: " << e.what();
            return absl::NotFoundError("python error");
        } catch (std::exception& e) {
            LOG(ERROR) << "PythonBackendCalculator::Process - Error: " << e.what();
            return absl::NotFoundError("generic error");
        }
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(PythonExecutorCalculator);
}  // namespace mediapipe