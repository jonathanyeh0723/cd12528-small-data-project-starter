# Starter code for Part 1 of the Small Data Solutions Project
# 

#Set up image data for train and test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Set up Transforms (train, val, and test)
#<<<YOUR CODE HERE>>>
# Define the transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

#Set up DataLoaders (train, val, and test)
batch_size = 10
num_workers = 4

#<<<YOUR CODE HERE>>>

# Load the datasets
data_dir = 'cd12528-small-data-project-starter/starter_code/part1-transfer/imagedata-50/'
train_dataset = datasets.ImageFolder(data_dir+'train', transform=train_transforms)
val_dataset = datasets.ImageFolder(data_dir+'val', transform=val_test_transforms)
test_dataset = datasets.ImageFolder(data_dir+'test', transform=val_test_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#hint, create a variable that contains the class_names. You can get them from the ImageFolder

# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories

#<<<YOUR CODE HERE>>>

# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

#<<<YOUR CODE HERE>>>
#device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")

#pretrained = models.vgg16(weights='DEFAULT').to(device)
pretrained = models.vgg16(weights='DEFAULT')

for param in pretrained.parameters():
    param.requires_grad = False

num_ftrs = pretrained.classifier[6].in_features
num_ftrs

pretrained.classifier[6] = nn.Linear(num_ftrs, 3)
model = pretrained

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
num_epochs = 5
# When you have all the parameters in place, uncomment these to use the functions imported above
def main():
   trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
   test_model(test_loader, trained_model, class_names='imagedata-50')

if __name__ == '__main__':
    main()
    print("done")
