#
# Copyright (c) 2022 Intel Corporation
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
import sys
sys.path.append("../../../../demos/common/python")

import numpy as np
import classes
import datetime
import argparse
from client_utils import print_statistics, prepare_certs
import tritonclient.grpc as grpcclient


if __name__ == '__main__':
    triton_client = grpcclient.InferenceServerClient(
                url='localhost:9000',
                verbose=True)

    inputs = []
    inputs.append(grpcclient.InferInput('in', (1,10), "FP32"))

    inputs[0].set_data_from_numpy(np.ones((1,10), dtype=np.float32))
    results = triton_client.infer(
        model_name='python_backend',
        inputs=inputs)

    print(results.as_numpy('out'))
