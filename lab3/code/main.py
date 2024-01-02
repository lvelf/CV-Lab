import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from Dataset import Dataset
from Net import *
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--description', type=str, default='Normal running',help='running description')
parser.add_argument('--nepochs', type=int, default=30,help='running epochs')
parser.add_argument('--model_type', type=str, default='Net',help='running epochs')
parser.add_argument('--patience', type=int, default=60,help='running epochs')

args = parser.parse_args()

description = args.description
nepochs = args.nepochs
model_type = args.model_type
patience_epoch = args.patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader, test_loader = Dataset()

print("now device is {}".format(device))

Net_type = 0


if model_type == 'Net':
    print('Net')
    model = Net()
elif model_type == 'SiameseNet':
    print('SiameseNet')
    model = SiameseNet()
else:
    model = SiameseNet_Residual()

model = model.to(device)
print(next(model.parameters()).device)
#import pdb; pdb.set_trace()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# losses record
train_losses = []
test_losses = []
test_accuracy = []
mini_batch_train_losses = []
mini_batch_test_losses = []
mini_batch_test_accuracy = []

max_test_accuracy = 0.0


for epoch in range(nepochs):
    running_loss = 0.0

    
    avg_train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
        inputs, labels = inputs.to(device), labels.to(device)

        #print(inputs.shape)
        #import pdb; pdb.set_trace()
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        # losses
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
        
    avg_train_loss = running_loss / len(train_loader)
    
    train_losses.append(avg_train_loss)
    
    running_loss = 0.0
        
    # test losses
    
    total_test = 0 
    correct_test = 0
    with torch.no_grad():
    
        for i, data in enumerate(test_loader, 0):
        
            inputs, labels = data
        
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            predicted = (outputs.data > 0.5).float()
            
            total_test += labels.size(0)

            correct_test += (predicted == labels).sum().item()
        
        avg_test_loss = running_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy.append(100.0000 * correct_test / total_test)
        
        if 100.0000 * correct_test / total_test > max_test_accuracy:
            max_test_accuracy =  100.0000 * correct_test / total_test
            torch.save(model.state_dict(), f'./models/best_model_{description}')
            print("saving max accuracy")
        
        if 100.0000 * correct_test / total_test < max_test_accuracy and epoch >= patience_epoch:
            print("early_stop")
            break
    
    running_loss = 0.0
    

        
total = 0 
correct = 0

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# load best model
if model_type == 'Net':
    print('Net')
    model = Net()
elif model_type == 'SiameseNet':
    print('SiameseNet')
    model = SiameseNet()
else:
    print('SiameseNet_Residual')
    model = SiameseNet_Residual()
    
model = model.to(device)
model.load_state_dict(torch.load(f'./models/best_model_{description}'))

with torch.no_grad():
    
    for i, data in enumerate(test_loader, 0):
        
        inputs, labels = data
        
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        predicted = (outputs.data > 0.5).float()

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


    
print(description)
print("Now Net type is {}".format(Net_type))
print('Accuracy on test data: %.4f %%' % (100 * correct / total))


total = 0 
correct = 0


with torch.no_grad():
    
    for i, data in enumerate(train_loader, 0):
        
        inputs, labels = data
        
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        predicted = (outputs.data > 0.5).float()

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print('Accuracy on train data: %.4f %%' % (100 * correct / total))

# draw losses
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig(f'./pictures/losses_and_accuracy_per_class_{description}.png')
#plt.show()

# draw test_accuracy
plt.figure(figsize=(8, 4))
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Over Epochs')
plt.legend()
plt.savefig(f'./pictures/test_accuracy_per_epoch_{description}.png')
#plt.show()
print("completing")

"""
# draw mini batch losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(mini_batch_train_losses, label='Training Loss')
plt.plot(mini_batch_test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mini batch Training and Test Loss')
plt.legend()
plt.savefig(f'./pictures/mini_batch_losses_{Net_type}.png')
"""