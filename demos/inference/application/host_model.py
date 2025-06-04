import logging
import uvicorn

from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline
from transformers import Pipeline, AutoTokenizer
from pydantic_settings import BaseSettings
from pydantic import Field, BaseModel
from datasets import load_dataset

# Create a logger from the global logger provider.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("host-model-for-inferencing")
g_inf_pipeline: Pipeline = None

inference_result_map = {"LABEL_1": 1, "LABEL_0": 0}


def get_inferencing_pipeline(model_path: str):
    global g_inf_pipeline
    if g_inf_pipeline is not None:
        return g_inf_pipeline
    logger.info(f"Loading model from path {model_path}.")
    model = ORTModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info(f"Hosting model for inference under pipeline for 'text-classification'")
    g_inf_pipeline = pipeline("text-classification", model=model, accelerator="ort", tokenizer=tokenizer)
    return g_inf_pipeline


def map_results(inference_result):
    val = inference_result[0]["label"]
    logger.debug(f"Mapping inference result {val} to label")
    if val in inference_result_map:
        return inference_result_map[val]
    return "Unknown"


class AppSettings(BaseSettings, cli_parse_args=True):
    model_path: str = Field(alias="model-path")
    data_path: str = Field(alias="data-path")
    application_port: int = Field(alias="application-port", default=8000)


class Data(BaseModel):
    data: str


settings = AppSettings()
app = FastAPI()

@app.post("/infer")
async def infer(data: Data):
    return map_results(get_inferencing_pipeline(settings.model_path)(data.data))

def data(dataset):
    for key in dataset.shuffle():
        yield key["text"], key["label"]

# TODO: This API is currently broken inside C-ACI as default scratch space is
# not writeable. Need to add configuration option to pass in the scratch space.
def do_check(splitName: str):
    dataset = load_dataset(settings.data_path, split=splitName)
    success_count = 0
    for key in data(dataset):
        inferred_result = map_results(get_inferencing_pipeline(settings.model_path)(key[0]))
        expected_result = key[1]
        if inferred_result == expected_result:
            success_count += 1
        logger.debug(
            f"Inference Result: {inferred_result}. Expected result: {expected_result}"
        )

    success_percentage = (success_count * 100.0) / dataset.num_rows

    result_map = {
        "Total rows in dataset" : dataset.num_rows,
        "Total number of succesful predictions" : success_count,
        "Success percentage" : success_percentage
    }

    return result_map

@app.get("/check/{splitName}")
def check_test(splitName: str):
    return do_check(splitName)

def main():
    get_inferencing_pipeline(settings.model_path)
    uvicorn.run(
        "host_model:app",
        host="0.0.0.0",
        port=settings.application_port,
        log_level="info",
    )

if __name__ == "__main__":
    main()
