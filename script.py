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

"""
outputs = None

class OvmsPythonModel:

    def initialize(self):
        # This method will be called once during initialization. 
        ...
        return None

    def execute(self, inputs: list) -> list:
        # This method will be called for every request.
        # It expects a list of inputs (our custom python objects) 
        # and returns list of outputs (also our custom python objects)
        ...
        return outputs
"""