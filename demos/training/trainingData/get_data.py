from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from pydantic_settings import BaseSettings
from pydantic import Field

class AppSettings(BaseSettings, cli_parse_args=True):
    data_path: str = Field(alias="data-path")

settings = AppSettings()

# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# When we run this code for the first time, the CIFAR10 train dataset will be downloaded locally. 
CIFAR10(root=settings.data_path,train=True,transform=transformations,download=True)