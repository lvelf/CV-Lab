import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from Dataset import Dataset
from Net import *
from Net_test import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader, test_loader = Dataset()

print("now device is {}".format(device))



model = Net()
model = model.to(device)
print(next(model.parameters()).device)
#import pdb; pdb.set_trace()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
    
    running_loss = 0.0
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
    


print('Accuracy on test data: %.4f %%' % (100 * correct / total))