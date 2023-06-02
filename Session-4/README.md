

## Overview
This code provides an implementation of a convolutional neural network (CNN) using PyTorch. The CNN is trained on the MNIST dataset for image classification. The code includes data processing, model definition, training loop, and visualization of the training and test data.

## Prerequisites
To run this code, you need the following dependencies:

- Python 3.9
- PyTorch
- torchvision
- matplotlib
- tqdm
- torchsummary

## Code Structure
The code is organized into several classes and functions to handle different aspects of the CNN training process.

## Model.py 
The Model.py file contains following classes and functions which are related to the model.

### `Net` Class
This class defines the architecture of the neural network. It consists of convolutional layers (`conv1`, `conv2`, `conv3`, `conv4`) followed by fully connected layers (`fc1`, `fc2`). The `forward` method defines the forward pass of the network.

### `model_summary` Function
This function generates a summary of the model architecture using the `torchsummary` library.

## Utils.py

### `DataProcessor` Class
This class handles data processing tasks. The `define_data_transforms` method defines the data transformations to be applied to the training and test data. The `download_dataset` method downloads the specified dataset (MNIST, FashionMNIST, CIFAR10) and applies the data transformations. The `define_data_loaders` method creates data loaders for the training and test data.

### `DataViewer` Class
This class provides methods for visualizing the training and test data. The `plot_train_data` method plots a batch of training data samples. The `plot_test_data` method plots a batch of test data samples.

### `check_cuda` Function
This function checks if CUDA is available for GPU acceleration.

### Training Loop
The code includes a training loop implemented in the `TrainLoop` class. The `train_model` method performs the training of the model using the specified optimizer. The `test_model` method evaluates the model on the test data. The `plot_graphs` method plots the accuracy and loss graphs during training.

### `GetCorrectPredCount` Function
This function calculates the number of correct predictions by comparing the predicted labels with the true labels.

### `get_device` Function
This function returns the device that will be used for computation (either 'cuda' if available, or 'cpu').


## Usage
1. Install the required dependencies listed in the "Prerequisites" section.
2. Import the necessary modules and classes from the code.
3. Create instances of the `DataProcessor`, `DataViewer`, and `TrainLoop` classes.
4. Check if CUDA is available for GPU acceleration.
5. Download the dataset and define data loaders using the `DataProcessor` class.
6. Initialize the model using the `Net` class and move it to the appropriate device.
7. Generate a summary of the model architecture using the `model_summary` function.
8. Define the training parameters such as the optimizer, scheduler, and number of epochs.
9. Start the training loop by iterating over the specified number of epochs.
10. During each epoch, call the `train_model` and `test_model` methods of the `TrainLoop` class.
11. Adjust the learning rate using the scheduler.
12. After training, plot the accuracy and loss graphs using the `plot_graphs` method of the `TrainLoop` class.

## Example
Here is an example of how to use the code:

```python
# Import the necessary modules and classes
import torch
import torch.nn as nn
import torch.optim as optim
from model import Net
from model import model_summary
from utils import TrainLoop
from utils import DataProcessor
from utils import DataViewer
from utils import check_cuda

# Create instances of the DataProcessor, DataViewer, and TrainLoop classes
dataprocessor = DataProcessor()
dataviewer = DataViewer()
trainingloop = TrainLoop()

# Check if CUDA is available
is_cuda_available = check_cuda()
device = torch.device("cuda" if is_cuda_available else "cpu")

# Download the dataset and define data loaders
data_transforms = dataprocessor.define_data_transforms()
train_data, test_data = dataprocessor.download_dataset(dataset_name="MNIST", data_transforms=data_transforms)
batch_size = 512
train_loader, test_loader = dataprocessor.define_data_loaders(batch_size, train_data, test_data)

# Initialize the model
model = Net().to(device)

# Generate a summary of the model architecture
model_summary = model_summary(model)
print(model_summary)

# Define the training parameters
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
num_epochs = 20

# Start the training loop
for epoch in range(1, num_epochs+1):
    print(f'Epoch {epoch}')
    trainingloop.train_model(model, device, train_loader, optimizer)
    trainingloop.test_model(model, device, test_loader)
    scheduler.step()

# Plot accuracy and loss graphs
trainingloop.plot_graphs()
```

