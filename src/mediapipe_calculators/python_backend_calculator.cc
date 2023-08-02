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
#include <openvino/openvino.hpp>

#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe {

class PythonBackendCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(ERROR) << "PythonBackendCalculator::GetContract";
        cc->Inputs().Index(0).Set<ov::Tensor>();
        cc->Outputs().Index(0).Set<ov::Tensor>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonBackendCalculator::Close";
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonBackendCalculator::Open";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(ERROR) << "PythonBackendCalculator::Process";

        ov::Tensor in_tensor = cc->Inputs().Index(0).Get<ov::Tensor>();

        auto* out_tensor = new ov::Tensor(in_tensor.get_element_type(), in_tensor.get_shape());
        std::memcpy(out_tensor->data(), in_tensor.data(), in_tensor.get_byte_size());

        for (int i = 0; i < 10; i++) {
            float* pV = ((float*)out_tensor->data()) + i;
            *pV += 1.0f;
        }

        cc->Outputs().Index(0).Add(out_tensor, cc->InputTimestamp());

        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(PythonBackendCalculator);

}  // namespace mediapipe
