#
# Copyright (c) 2023 Intel Corporation
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
import time
import tritonclient.grpc as grpcclient

channel_args = [
    # Do not drop the connection for long workloads
    ("grpc.http2.max_pings_without_data", 0),
]
client = grpcclient.InferenceServerClient("localhost:11339", channel_args=channel_args)
def callback(result, error):
   if error:
       raise error
   timestamp = result.get_response().parameters['OVMS_MP_TIMESTAMP'].int64_param  # optional
   print(timestamp, result.get_response().as_numpy('OUTPUT').tobytes().decode())
client.start_stream(callback=callback)
text = "Describe the state of the healthcare industry in the United States in max 2 sentences"
print(f"Question:\n{text}\n")
data = text.encode()
infer_input = grpcclient.InferInput("pre_prompt", [len(data)], "BYTES")
infer_input._raw_content = data

# send 100 requests and do not wait for responses
# for cycle we should edit it to send only 1
for i in range(100):
    client.async_stream_infer(
       model_name="python_model",
       inputs=[infer_input])  # optional

# wait 1hour for ctrl + c
time.sleep(60*60)

client.stop_stream()
