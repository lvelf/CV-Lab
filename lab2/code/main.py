import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from Dataset import Dataset
from Net import *
import matplotlib.pyplot as plt
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader, test_loader = Dataset()

print("now device is {}".format(device))

Net_type = 0



model = Net()
model = model.to(device)
print(next(model.parameters()).device)
#import pdb; pdb.set_trace()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# losses record
train_losses = []
test_losses = []
test_accuracy = []
mini_batch_train_losses = []
mini_batch_test_losses = []
mini_batch_test_accuracy = []

max_test_accuracy = 0.0


for epoch in range(10):
    running_loss = 0.0

    
    avg_train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        # losses
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
        """
        mini_batch_train_losses.append(running_loss / 100)
        
        # mini_ batch test losses
        total_test = 0 
        correct_test = 0
        test_running_loss = 0
        with torch.no_grad():
    
            for i, data in enumerate(test_loader, 0):
        
                inputs, labels = data
        
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
            
                loss = criterion(outputs, labels)
            
                test_running_loss += loss.item()
            
                _, predicted = torch.max(outputs.data, 1)
            
                total_test += labels.size(0)

                correct_test += (predicted == labels).sum().item()
        
        avg_test_loss = running_loss / len(test_loader)
        mini_batch_test_losses.append(avg_test_loss)
        mini_batch_test_accuracy.append(100.0000 * correct_test / total_test)
        """
        
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
            
            _, predicted = torch.max(outputs.data, 1)
            
            total_test += labels.size(0)

            correct_test += (predicted == labels).sum().item()
        
        avg_test_loss = running_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy.append(100.0000 * correct_test / total_test)
        
        if 100.0000 * correct_test / total_test > max_test_accuracy:
            max_test_accuracy =  100.0000 * correct_test / total_test
            torch.save(model.state_dict(), f'./models/best_model_{Net_type}')
            print("saving max accuracy")
        
        if 100.0000 * correct_test / total_test < max_test_accuracy:
            print("early_stop")
            break
    
    running_loss = 0.0

        
total = 0 
correct = 0

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# load best model
model = Net().to(device)
model.load_state_dict(torch.load(f'./models/best_model_{Net_type}'))

with torch.no_grad():
    
    for i, data in enumerate(test_loader, 0):
        
        inputs, labels = data
        
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        total += labels.size(0)

        correct += (predicted == labels).sum().item()
    

print("Now Net type is {}".format(Net_type))
print('Accuracy on test data: %.4f %%' % (100 * correct / total))

total = 0 
correct = 0


with torch.no_grad():
    
    for i, data in enumerate(test_loader, 0):
        
        inputs, labels = data
        
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print('Accuracy on train data: %.4f %%' % (100 * correct / total))

# draw losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

# draw every type accuracy
plt.subplot(1, 2, 2)
class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(10)]
plt.bar(range(10), class_accuracy)
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Class on Test Set')

plt.tight_layout()
plt.savefig(f'./pictures/losses_and_accuracy_per_class_{Net_type}.png')
#plt.show()

# draw test_accuracy
plt.figure(figsize=(8, 4))
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Over Epochs')
plt.legend()
plt.savefig(f'./pictures/test_accuracy_per_epoch_{Net_type}.png')
#plt.show()

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