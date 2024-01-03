import torch
from torchvision import datasets, transforms
import random
from torch.utils.data import TensorDataset, DataLoader
import os


def Dataset():
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_data_loader, repeatition = get_transferred_data(train_dataset, 64, True, 'train')
    
    test_data_loader, repeatition = get_transferred_data(test_dataset, 64, False, 'test', repeatition = repeatition)
    
    
    return train_data_loader, test_data_loader



def get_transferred_data(train_dataset, batch_size, shuffle, set_name, repeatition = None):
    
    dataset_path = f'./Dataset/{set_name}_data_set'
    labels_path = f'./Dataset/{set_name}_labels'
    
    if os.path.exists(dataset_path) and os.path.exists(labels_path):
        new_train_samples = torch.load(dataset_path)
        new_train_labels = torch.load(labels_path)
        train_data_set = torch.utils.data.TensorDataset(torch.stack(new_train_samples), torch.tensor(new_train_labels))

        train_data_loader = DataLoader(train_data_set, batch_size = batch_size, shuffle = shuffle)
        return train_data_loader, None
    
    # randomly get 10% train data
    train_samples = train_dataset.data
    num_samples = int(len(train_samples) * 0.1)
    random_index = torch.randperm(train_samples.size(0))
    
    
    train_samples = train_samples[random_index[:num_samples] ]
    print(num_samples)
    print(len(train_samples))
    
    #get labels
    train_labels = train_dataset.targets[random_index[:num_samples]]
    
    new_train_samples = []
    new_train_labels = []
    
    
    train_number = 0
    train_set = set()
        
        
    while(train_number < 100000):
        i = random.randint(0, len(train_samples) - 1)
        j = random.randint(0, len(train_samples) - 1)
        
        if i == j or (i,j) in train_set:
            continue
        
        if repeatition != None:
            if (i,j) in repeatition:
                continue

        if train_labels[i] == train_labels[j]:
            new_label = 1
        else:
            new_label = 0
                
        combined_sample = torch.cat((train_samples[i], train_samples[j]), dim = 1)
            
        new_train_samples.append(combined_sample)
        new_train_labels.append(new_label)
        
        train_number += 1
        train_set.add((i, j))
        train_set.add((j, i))
            
            
    for i in range(len(new_train_samples)):
        new_train_samples[i] = new_train_samples[i].unsqueeze(0).float()
    
    new_train_labels = torch.tensor(new_train_labels, dtype=torch.float).unsqueeze(1)
    
    
    train_data_set = torch.utils.data.TensorDataset(torch.stack(new_train_samples), torch.tensor(new_train_labels))

    train_data_loader = DataLoader(train_data_set, batch_size = batch_size, shuffle = shuffle)
    
    
    os.makedirs('./Dataset', exist_ok=True) 
    torch.save(new_train_samples, dataset_path)
    torch.save(new_train_labels, labels_path)
    
    
    num_onses = torch.sum(new_train_labels == 1).item()
            
    if repeatition != None:
        return train_data_loader, None
    
    if repeatition == None:
        return train_data_loader, train_set


train_data_loader, test_data_loader = Dataset()
