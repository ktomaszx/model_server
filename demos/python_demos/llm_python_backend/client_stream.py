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
import tritonclient.grpc as grpcclient

def callback(result, error):
   if error:
       raise error
   # ... process result
   timestamp = result.get_response().parameters['OVMS_MP_TIMESTAMP'].int64_param  # optional
   print(result.as_numpy('OUTPUT').tobytes().decode(), flush=True, end='')

channel_args = [
    # Do not drop the connection for long workloads
    ("grpc.http2.max_pings_without_data", 0),
]
client = grpcclient.InferenceServerClient("localhost:11339", channel_args=channel_args)
client.start_stream(callback=callback)
text = "Describe current state of United States healthcare"
print(f"Question:\n{text}\n")
data = text.encode()
infer_input = grpcclient.InferInput("pre_prompt", [len(data)], "BYTES")
infer_input._raw_content = data
#results = client.infer("python_model", [infer_input], client_timeout=60*60)  # 1 hour
#print(f"Completion:\n{results.as_numpy('OUTPUT').tobytes().decode()}\n")
for _ in range(1):  # re-use opened stream
    client.async_stream_infer(
       model_name="python_model",
       inputs=[infer_input])

#client.stop_stream()
