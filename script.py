import numpy as np

class OvmsPythonModel:

    def initialize(self):
        print("Python model initialized")
        return None

    def execute(self, inputs: dict) -> dict:
        input_arr = inputs["input"]
        output_arr = input_arr + 2
        output_dict = {"output": output_arr}
        return output_dict


def executex(inputs:dict) -> dict:
    """
    from optimum.intel import OVStableDiffusionPipeline

    model_id = "echarlaix/stable-diffusion-v1-5-openvino"
    stable_diffusion = OVStableDiffusionPipeline.from_pretrained(model_id)
    prompt = "sailing ship in storm by Rembrandt"
    images = stable_diffusion(prompt).images
    """
    #from optimum.intel import OVModelForSequenceClassification
    from transformers import AutoTokenizer, pipeline, DistilBertForSequenceClassification

    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    model = DistilBertForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    cls_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    phrases = ["Analysis for those sentences are done in Python interpreter, by Torch runtime, via Transformers library, with model from HF. How awesome is that?",
               "Python custom nodes will revolutionize OVMS.", 
               "I tried to run it with OV optimized model, but failed unfortunately. Perhaps OpenVINO in OVMS and OpenVINO in pip package are conflicting?", 
               "Also multithreading with Python embedded interpreter is not an easy thing and might cause multiple problems, so there's still a lot of work ahead of us."]
    for phrase in phrases:
        print("\n")
        print(phrase)
        print(cls_pipe(phrase))
        print("\n")
    return {"output": inputs["input"]}

#if __name__ == "__main__":
#    execute({"input": [1,2,3,4,5]})
