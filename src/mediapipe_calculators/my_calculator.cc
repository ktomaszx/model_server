#include <iostream>
#include <openvino/openvino.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop

namespace mediapipe {

class MyCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Index(0).Set<ov::Tensor>();
        cc->Outputs().Index(0).Set<ov::Tensor>();
        cc->InputSidePackets().Index(0).Set<int>();  // TODO: GrpcWriterSender
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        int a = cc->InputSidePackets().Index(0).Get<int>();  // TODO: GrpcWriterSender
        std::cout << "-------======------ Processing MyCalculator, value=" << a << std::endl;
        cc->Outputs().Index(0).Add(
            new ov::Tensor(
                ov::element::Type_t::f32,
                ov::Shape{1,1}),
            cc->InputTimestamp());
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(MyCalculator);

}  // namespace mediapipe
