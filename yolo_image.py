#####################################################
# yolo_image.py
# training and validating YOLO on BEE4 image dataset.
# bugs to vladimir kulyukin, chris allred on canvas
#####################################################

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from sys import platform
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim

### change the number of epochs; I set it to 3.
CONFIG_DICT = {
    'model': 'yolo_image.pth',
    'plot':  'yolo_plot.png',
    'debug': True,
    'INIT_LR': 1e-3,
    'BATCH_SIZE': 64,
    'EPOCHS': 75,
    'TRAIN_SPLIT': 0.75,
}

# define training hyperparameters
INIT_LR = CONFIG_DICT['INIT_LR']
BATCH_SIZE = CONFIG_DICT['BATCH_SIZE']
EPOCHS = CONFIG_DICT['EPOCHS']
TRAIN_SPLIT = CONFIG_DICT['TRAIN_SPLIT']
VAL_SPLIT = 1 - TRAIN_SPLIT
# set the device we will be using to train the model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0' if torch.cuda.is_available() else ('mps' if platform == "darwin" else 'cpu')
print(f'<DBG> device: {device}')

# load the bee dataset
# Define transformations for image preprocessing
# Define your transformation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # Randomly flip images horizontally
    transforms.RandomVerticalFlip(), # Randomly flip images vertically
    transforms.RandomRotation(10), # Randomly rotate images by up to 10␣degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Randomly adjust brightness, contrast, saturation, and hue
    transforms.Resize((32, 32)), # Resize the images to a consistent size
    transforms.ToTensor(), # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize the␣ image pixel values
])

# load the BEE4 dataset
print('<DBG> loading BEE4 train and test datasets...')
trainData = ImageFolder(root='data/BEE4/train', transform=transform)
testData  = ImageFolder(root='data/BEE4/valid', transform=transform)

# calculate the train/validation split
print('<DBG> generating the train/validation split...')
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = len(trainData) - numTrainSamples
# numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData,
[numTrainSamples, numValSamples],
generator=torch.Generator().manual_seed(13))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader   = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader  = DataLoader(testData, batch_size=BATCH_SIZE)

class YOLO_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(YOLO_Classifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# Define the number of classes for your dataset
num_classes = 2 # This assumes the number of classes is the same as the number␣of classes in your dataset.
# Create an instance of the YOLOClassifier
model = YOLO_Classifier(num_classes).to(device)

# Define the loss function (e.g., CrossEntropyLoss) and optimizer (e.g., SGD or␣Adam)
lossFn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())
# opt = torch.optim.SGD(model.parameters(), lr=0.01)
# Training loop, validation loop, and testing loop can be implemented using the␣defined model, DataLoader, loss, and optimizer.

# H is a dictionary to store training history
H = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
# we'll measure how long training is going to take
print('<DBG> training YOLO...')
startTime = time.time()

for e in range(0, EPOCHS):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    total_train_loss = 0
    total_val_loss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    train_correct = 0
    val_correct = 0
    # loop over the training set
    for (x, y) in trainDataLoader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        total_train_loss += loss
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in valDataLoader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # make the predictions and calculate the validation loss
            pred = model(x)
            total_val_loss += lossFn(pred, y)
            # calculate the number of correct predictions
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    # Comput the stats:
    # calculate the average training and validation loss
    avg_train_loss = total_train_loss / trainSteps
    avg_val_loss = total_val_loss / valSteps
    # calculate the training and validation accuracy
    train_correct = train_correct / len(trainDataLoader.dataset)
    val_correct = val_correct / len(valDataLoader.dataset)

    # update training history
    H['train_loss'].append(float(avg_train_loss.cpu().detach().numpy()))
    H['train_acc'].append(train_correct)
    H['val_loss'].append(float(avg_val_loss.cpu().detach().numpy()))
    H['val_acc'].append(val_correct)
    # print the model training and validation information
    print('<DBG> EPOCH: {}/{}'.format(e + 1, EPOCHS))
    print('Train loss: {:.4f}, Train accuracy: {:.4f}'.format(avg_train_loss, train_correct))
    print('Val loss: {:.4f}, Val accuracy: {:.4f}\n'.format(avg_val_loss, val_correct))

#finish measuring how long training took
endTime = time.time()
print('<DBG> total time taken to train the model: {:.2f}s'.format(endTime - startTime))

### =============== TESTING YOLO =======================

sample_size = 10
print('<DBG> evaluating network...')
rand_idx = np.random.randint(0, len(testDataLoader), size=sample_size)
print('<DBG> rand_idx: ', rand_idx)
image_idx =0
# turn off autograd for testing evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    # initialize a list to store our predictions
    preds = []
    # loop over the test set
    for i,(x, y) in enumerate(testDataLoader):
        # send the input to the device
        x = x.to(device)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
        # tital_srt = testData.dataset.samples[i][0]
        if i in rand_idx:
            # Reverse the normalization for display
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.5, 0.5, 0.5])
            sample_image = (x[0].permute(1, 2, 0).cpu() * std) + mean
            #ax[image_idx].imshow(sample_image)
            title_str = f'Predicted: {pred.argmax(axis=1).cpu().numpy()[0]}\n Actual: {y.cpu().numpy()[0]}'
            print(title_str)

# let's generate a classification report
print(classification_report(testData.targets, np.array(preds),
                            target_names=testData.classes))

# persist the model to disk
torch.save(model, CONFIG_DICT['model'])
