# Llama demo with python node {#ovms_demo_python_llama}

This demo shows how to take advantage of OpenVINO Model Server to generate content remotely with LLM models. 
The server is running here the user provided python code in a MediaPipe Python calculator. We manage the generation with Hugging Faces pipeline and OpenVINO Runtime as the execution engine.
The generation cycles are configured in the MediaPipe graph. Two use cases are possible:
- with unary calls - when the client is sending a single prompt to the graph and receives a complete generated response
- with gRPC streaming - when the client is sending a single prompt the graph and receives a stream of responses

The unary calls are simpler but the response might be sometimes slow when many cycles are needed on the server side

The gRPC stream is a great feature when more interactive approach is needed allowing the user to read the response as they are getting generated.

This demo is presents the use case with `red-pajama-3b-chat`` model but the included python scripts are prepared for several other LLM  models like `Llama-2-7b-chat-hf` and TBD.

## Requirements

Linux host with a docker engine installed and adequate available RAM to load the model or Intel GPU card.
This demo was tested on a host with Intel(R) Xeon(R) Gold 6430 and Flex170 GPU card.

## Build image

Building the image with all required python dependencies is required. Follow the commands:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```
It will create an image called `openvino/model_server:py`

## Download model

We are going to use [red-pajama-3b-chat](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1) model in this scenario.
Download the model using `download_model.py` script:

```bash
cd demos/python_demos/llm_python_backend
pip install -r requirements.txt
python3 download_model.py
```
The model will appear in `./model` directory.

## Model quantization - optional
TBD

## Deploy OpenVINO Model Server with the Python calculator and CPU execution

Mount the `./model` directory with the model.  
Mount the `./servable` which contains:
- `model.py` and `config.py` - python scripts which are required for execution and use [Hugging Face](https://huggingface.co/) utilities with [optimum-intel](https://github.com/huggingface/optimum-intel) acceleration.
- `config.json` - which defines which servables should be loaded
- `graph.pbtxt` - which defines MediaPipe graph containing python calculator

```bash
docker run -it --rm -p 9000:9000 -v ${PWD}/servable:/workspace -v ${PWD}/model:/model openvino/model_server:py --config_path /workspace/config.json --port 9000
```

## Deploy OpenVINO Model Server with the Python calculator and GPU execution

Deployment of the service on GPU is very similar to CPU. The only difference is that the GPU device needs to be passed to the container and a proper security context is required with access to the GPU device.

```bash
docker run -it --rm --device /dev/dri --group_add=$(stat -c "%g" /dev/dri/render*) -u $(id -u) -p 9000:9000 -v ${PWD}/servable:/workspace -v ${PWD}/model:/model openvino/model_server:py --config_path /workspace/config.json --port 9000
```
Note that the python initialization method in the calculator has the `AUTO` device configured so it select GPU if present or CPU overwise.
You might also tune there the OpenVINO config for like with enforcing a specific execution precision `{"INFERENCE_PRECISION_HINT": "f32"}`.


## Requesting the LLM with unary calls

The client script contains hardcoded prompt:
```
Describe the state of the healthcare industry in the United States in max 2 sentences
```

Run time client
```bash
python3 client.py --url localhost:9000
```

Example output:
```bash
Question:
Describe the state of the healthcare industry in the United States in max 2 sentences

Completion:
 Many jobs in the health care industry are experiencing long-term shortages due to a lack of workers, while other areas face overwhelming stress and strain.  Due to COVID-19 many more people look for quality medical services closer to home so hospitals have seen record levels of admissions over the last year.
```

## Requesting the LLM with gRPC streaming

The python code deployed in the model server is very similar with the unary calls and gRPC streams. There is required just one adjustment to ensure the generation cycle includes sending the partial results.
The execute method should end with the `yield` operator instead of `return`:

```bash
sed ...
```

After such adjustment start OpenVINO Model Server like before:
```bash
docker run -it --rm -p 9000:9000 -v ${PWD}/servable:/workspace -v ${PWD}/model:/model openvino/model_server:py --config_path /workspace/config.json --port 9000
```

Now run this [client](stream_client.py) to send the prompt and read the stream of responses. Each reasons in handled by the callback function defined in the stream initialization.

