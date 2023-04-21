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
#include <iostream>
#include <sstream>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "../ovms.h"           // NOLINT
#include "../stringutils.hpp"  // TODO dispose
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "modelapiovmsadapter.hpp"
#include "modelapiovmsadapterwrapper.hpp"
#include "src/mediapipe_calculators/modelapiovmssessioncalculator.pb.h"
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {
#define MLOG(A) LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << A << std::endl;

using ovms::AdapterWrapper;
using ovms::OVMSInferenceAdapter;
using std::endl;

const std::string SESSION_TAG{"SESSION"};

class ModelAPISessionCalculator : public CalculatorBase {
    std::unique_ptr<AdapterWrapper> adapter;
    std::unordered_map<std::string, std::string> outputNameToTag;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        MLOG("Session GetContract start");
        RET_CHECK(cc->Inputs().GetTags().empty());
        RET_CHECK(cc->Outputs().GetTags().empty());
        cc->OutputSidePackets().Tag(SESSION_TAG.c_str()).Set<AdapterWrapper>();
        const auto& options = cc->Options<ModelAPIOVMSSessionCalculatorOptions>();
        RET_CHECK(!options.servable_name().empty());
        MLOG("Session GetContract middle");
        // TODO validate version from string
        // TODO validate service url format
        // this is for later support for remote server inference
        MLOG("Session GetContract end");
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        MLOG("Session Open start");
        for (CollectionItemId id = cc->Inputs().BeginId();
             id < cc->Inputs().EndId(); ++id) {
            if (!cc->Inputs().Get(id).Header().IsEmpty()) {
                cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
            }
        }
        if (cc->OutputSidePackets().NumEntries() != 0) {
            for (CollectionItemId id = cc->InputSidePackets().BeginId();
                 id < cc->InputSidePackets().EndId(); ++id) {
                cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
            }
        }
        cc->SetOffset(TimestampDiff(0));

        const auto& options = cc->Options<ModelAPIOVMSSessionCalculatorOptions>();
        const std::string& servableName = options.servable_name();
        const std::string& servableVersion = options.servable_version();
        auto session = std::make_unique<AdapterWrapper>(new OVMSInferenceAdapter(servableName));
        MLOG("Session create adapter");
        cc->OutputSidePackets().Tag(SESSION_TAG.c_str()).Set(Adopt(session.release()));
        MLOG("SessionOpen end");
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        MLOG("SessionProcess");
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(ModelAPISessionCalculator);
}  // namespace mediapipe