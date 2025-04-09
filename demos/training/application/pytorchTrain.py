import time
import os
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.autograd import Variable

from torch.utils.data import DataLoader

from pydantic_settings import BaseSettings
from pydantic import Field

import torch.nn as nn
import torch.nn.functional as F

class AppSettings(BaseSettings, cli_parse_args=True):
    inmodel_path: str = Field(alias="model-path")
    data_path: str = Field(alias="data-path")
    outmodel_path: str = Field(alias="out-path")

# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output

settings = AppSettings()

def loadData(data_path):

    # Loading and normalizing the data.
    # Define transformations for the training and test sets
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5,000 batches of images.
    batch_size = 10
    number_of_labels = 10

    print(f"Loading data from {data_path}")

    # Create an instance for training.
    train_set =CIFAR10(root=data_path,train=True,transform=transformations,download=False)

    # Create a loader for the training set which will read the data within batch size and put into memory.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    print("The number of images in a training set is: ", len(train_loader)*batch_size)

    # Create an instance for testing, note that train is set to False.
    test_set = CIFAR10(root=data_path, train=False, transform=transformations,download=False)

    # Create a loader for the test set which will read the data within batch size and put into memory.
    # Note that each shuffle is set to false for the test loader.
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print("The number of images in a test set is: ", len(test_loader)*batch_size)

    print("The number of batches per epoch is: ", len(train_loader))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader


# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy(model, test_loader):
    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images.to(device))
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(model, device, epoch, train_loader, test_loader):
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader, 0):

        # get the inputs
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        # zero the parameter gradients
        optimizer.zero_grad()
        # predict classes using images from the training set
        outputs = model(images)
        # compute the loss based on model output and real labels
        loss = loss_fn(outputs, labels)
        # backpropagate the loss
        loss.backward()
        # adjust parameters based on the calculated gradients
        optimizer.step()

        # Let's print statistics for every 1,000 images
        running_loss += loss.item()     # extract the loss value
        if i % 1000 == 999:
            # print every 1000 (twice per epoch)
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1000))
            # zero the loss
            running_loss = 0.0

    # Compute and print the average accuracy for this epoch when tested over all 10000 test images
    accuracy = testAccuracy(model, test_loader)
    print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))


def main():
    path = f"{settings.inmodel_path}/model.pth"
    outPath = f"{settings.outmodel_path}/model.pth"

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Instantiate a neural network model
    model = Network()

    # Load model from path
    if (os.path.exists(path)):
        print(f"Model file exists at {path}")

        # Load the state dict
        model.load_state_dict(torch.load(path))

        # Load the data
        train_loader, test_loader = loadData(settings.data_path)

        # Convert model parameters and buffers to CPU or Cuda
        model.to(device)

        start = time.time()
        # Train & loop over the dataset multiple times
        for epoch in range(3):
            train(model, device, epoch, train_loader, test_loader)
        print('Finished Training')

        # <code to time>
        end = time.time()
        print(f"Time taken to train was {end-start} seconds")

        # Save the models
        print(f"Saving trained model to {outPath}")
        torch.save(model.state_dict(), outPath)
    else:
        print(f"Model file {path} does not exist")
        exit(1)

if __name__ == "__main__":
    main()