from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import ray
from vllm import LLM, SamplingParams
from loguru import logger
import sys


# Define the configuration table
config_table = {
    "llama2": {
        "max_model_len": 2048,
        "id2score": {29900: "0", 29896: "1"}
    },
    "llama3": {
        "max_model_len": 8192,
        "id2score": {15: "0", 16: "1"}
    },
    "mistral": {
        "max_model_len": 2000,
        "id2score": {28734: "0", 28740: "1"}
    }
}

# Request and Response models
class InferenceRequest(BaseModel):
    input_data: List[str]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True

class InferenceResponse(BaseModel):
    outputs: List[str]

app = FastAPI()

def get_model_config(model_path: str):
    for key in config_table:
        if key in model_path.lower():
            logger.info(f"Using config for {key}")
            return config_table[key]
    return config_table["mistral"]


@app.post("/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    global llm
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            skip_special_tokens=request.skip_special_tokens
        )
        outputs = llm.generate(request.input_data, sampling_params)
        output_texts = [output.outputs[0].text for output in outputs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    return InferenceResponse(outputs=output_texts)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python vllm_server.py <port> <model_path>")
        sys.exit(1)
    port = int(sys.argv[1])
    model_path = sys.argv[2]


    config = get_model_config(model_path)
    try:
        llm = LLM(model=model_path, tokenizer_mode="auto", trust_remote_code=True, max_model_len=config["max_model_len"], gpu_memory_utilization=0.95)
        logger.info(f"Model {model_path} loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    # Start the server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
    
    
    
