class OvmsPythonModel:

    def initialize(self):
        print("Python model initialized")
        return None

    def execute(self): #, inputs: dict) -> dict:
        print("Python model execute")
        return None
        #input_arr = inputs["input"]
        #output_arr = input_arr + 2
        #output_dict = {"output": output_arr}
        #return output_dict


outputs = None

class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        # This method will be called once during initialization.
        # It expects keyword arguments. They will map node configuration from pbtxt including node options, node name etc.
        # Detailed spec to be provided. 
        ...
        return None

    def execute(self, inputs: list, kwargs: dict) -> list:
        # This method will be called for every request.
        # It expects a list of inputs (our custom python objects).
        # It also expects keyword arguments. They will be provided by the calculator to enable advanced processing and flow control.
        # Detailed spec to be provided. 
        #
        # It will returns list of outputs (also our custom python objects).
        ...
        return outputs

    def finalize(self, kwargs: dict):
        # This method will be called once during deinitialization. 
        # It expects keyword arguments. They will map node configuration from pbtxt including node options, node name etc.
        # Detailed spec to be provided. 
        ...
        return None

def load_model(xd): pass
import numpy as np

class OvmsPythonModel:

    def initialize(self):
        self.model = load_model("/ovms/models/llama")
        return None

    def execute(self, inputs: list) -> list:
        output = []
        input = np.frombuffer(inputs[0].data, dtype=inputs[0].datatype).reshape(inputs[0].shape)
        for sentence in input:
            output.append(self.model(sentence))
        return [np.array(output)]
    
client = 1
KServeRequest = 1
KServeInput = 1
import cv2

input_queries = np.array(["What is llama?", "Why do we need llama?", "Why not a horse?"])

request = KServeRequest()
input = KServeInput()

input.shape = list(input_queries.shape)
input.datatype = str(input_queries.dtype)
input.raw_input_contents = input_queries.tobytes()

request.inputs.append(input)
response = client.predict(request, "python_backend")

output = response.outputs[0]
# We could potentially provide utility here (ovmsclient with KServe support?)
results = np.frombuffer(output.raw_output_contents, output.datatype).reshape(output.shape)

for result in results:
    print(result)


# client.py
input.shape = [1,3,300,300]
input.datatype = "FLOAT32"
input.raw_input_contents = "<input_image_bytes>"

# script.py
def execute(self, inputs: list) -> list:
    output = []
    input = np.array(inputs[0])
    output = cv2.imread(input).resize(720,720).rotate(30).numpy()
    return [output]



# script.py
def execute(self, inputs: list) -> list:
    for i in range(20):
        inputs = self.model(inputs)
        yield inputs
