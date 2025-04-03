import logging
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from pydantic_settings import BaseSettings
from pydantic import Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("get_model")


class AppSettings(BaseSettings, cli_parse_args=True):
    output_path: str = Field(alias="output-path")


settings = AppSettings()
model_id = "distilbert/distilbert-base-uncased"

logging.info(f"Fetching model {model_id} from huggingface Hub.")
model = DistilBertForSequenceClassification.from_pretrained(model_id)
logging.info(f"Saving model to path {settings.output_path}")
model.save_pretrained(f"{settings.output_path}/pytorch")

tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
logging.info(f"Saving tokenizer to path {settings.output_path}")
tokenizer.save_pretrained(f"{settings.output_path}/pytorch")

model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
logging.info(f"Saving ONNX model to path {settings.output_path}")
model.save_pretrained(f"{settings.output_path}/onnx")

# TODO: Should we tune the model to give better inference results?
