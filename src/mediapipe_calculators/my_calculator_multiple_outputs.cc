#include <iostream>
#include <openvino/openvino.hpp>
#include <chrono>
#include <thread>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop

namespace mediapipe {

class MyMultipleOutputCalculator : public CalculatorBase {
    int iteration = 0;
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Index(0).Set<ov::Tensor>();
        cc->Inputs().Index(1).Set<ov::Tensor>();
        cc->Outputs().Index(0).Set<ov::Tensor>();
        cc->Outputs().Index(1).Set<ov::Tensor>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        if (iteration >= 5) {
            return absl::OkStatus();
        }

        int data_index = cc->Inputs().Index(0).IsEmpty() ? 1 : 0;
        ov::Tensor data = cc->Inputs().Index(data_index).Get<ov::Tensor>(); 

        ov::Tensor output1 = ov::Tensor(data.get_element_type(), data.get_shape());
        std::memcpy(output1.data(), data.data(), output1.get_byte_size());
        for (size_t i = 0; i < output1.get_byte_size() / sizeof(int64_t); i++) {
            reinterpret_cast<int64_t*>(output1.data())[i] += 1;
        }

        ov::Tensor output2 = ov::Tensor(data.get_element_type(), data.get_shape());
        std::memcpy(output2.data(), data.data(), output2.get_byte_size());
        for (size_t i = 0; i < output2.get_byte_size() / sizeof(int64_t); i++) {
            reinterpret_cast<int64_t*>(output2.data())[i] += 2;
        }
        // imitate workload
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(1000ms);

        iteration++;
        cc->Outputs().Index(0).Add(new ov::Tensor(output1), Timestamp(iteration));
        cc->Outputs().Index(1).Add(new ov::Tensor(output2), Timestamp(iteration));
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(MyMultipleOutputCalculator);

}  // namespace mediapipe
