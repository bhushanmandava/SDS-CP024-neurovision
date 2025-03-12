#!/usr/bin/env python
# coding: utf-8

# In[2]:



# In[17]:





# In[19]:


import os

folder_path_valid = '/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor/valid/labels'
folder_path = '/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor/train/labels'

# Lists to store the processed data
output_values = []
output_values_valid = []

# Function to read and process label files
def process_label_files(folder_path):
    output_values = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                content = file.read().strip()
                values = content.split()  # Split the string into individual elements
                try:
                    values = [float(value) for value in values]  # Convert each value to float
                    output_values.append(values)  # Append to the output list
                except ValueError:
                    print(f"Skipping file {filename}, invalid data found.")
    return output_values

# Process the training labels
output_values = process_label_files(folder_path)

# Process the validation labels
output_values_valid = process_label_files(folder_path_valid)

# Print the results
print("length of training ",  len(output_values))
print("length of valid labels:", len(output_values_valid))
y_train=[]
for i in output_values:
    y_train.append(i[0])
print(len(y_train)) 
y_test=[]
for i in output_values_valid:
    y_test.append(i[0])
print(len(y_test))


# In[20]:


image_filenames =os.listdir('/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor/train/images')
print(len(image_filenames))
image_filenames_valid =os.listdir('/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor/valid/images')
print(len(image_filenames_valid))


# In[21]:


import os
import shutil

# Paths
image_directory = '/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor/train/images'  # Update the image directory path
tumor_directory = '/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor/train/tumor'  # Path for tumor images
no_tumor_directory = '/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor/train/no_tumor'  # Path for no tumor images

# Assuming y_train is a list/array of binary values where 1 = tumor, 0 = no tumor
# y_train = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # Example binary list; replace with your actual y_train

# Example list of file names (these should match the images in your image directory)
# image_filenames = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg', 'image6.jpg', 'image7.jpg', 'image8.jpg', 'image9.jpg', 'image10.jpg', 'image11.jpg', 'image12.jpg', 'image13.jpg', 'image14.jpg', 'image15.jpg', 'image16.jpg', 'image17.jpg', 'image18.jpg', 'image19.jpg', 'image20.jpg']  # Example image names

# Create the destination directories if they don't exist
os.makedirs(tumor_directory, exist_ok=True)
os.makedirs(no_tumor_directory, exist_ok=True)

# Loop through the binary values in y_train and the corresponding image file names
for label, image_name in zip(y_train, image_filenames):
    # Get the full path of the image file
    image_path = os.path.join(image_directory, image_name)
    
    if label == 1:
        # Move file to tumor directory
        shutil.move(image_path, os.path.join(tumor_directory, image_name))
        print(f'Moved {image_name} to Tumor directory')
    elif label==0:
        # Move file to no_tumor directory
        shutil.move(image_path, os.path.join(no_tumor_directory, image_name))
        print(f'Moved {image_name} to No Tumor directory')
        

print("File organization complete!")


# In[22]:


import os
import shutil

# Paths
image_directory = '/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor/valid/images'  # Update the image directory path
tumor_directory = '/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor/valid/tumor'  # Path for tumor images
no_tumor_directory = '/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor/valid/no_tumor'  # Path for no tumor images

# Assuming y_train is a list/array of binary values where 1 = tumor, 0 = no tumor
# y_train = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # Example binary list; replace with your actual y_train

# Example list of file names (these should match the images in your image directory)
# image_filenames = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg', 'image6.jpg', 'image7.jpg', 'image8.jpg', 'image9.jpg', 'image10.jpg', 'image11.jpg', 'image12.jpg', 'image13.jpg', 'image14.jpg', 'image15.jpg', 'image16.jpg', 'image17.jpg', 'image18.jpg', 'image19.jpg', 'image20.jpg']  # Example image names

# Create the destination directories if they don't exist
os.makedirs(tumor_directory, exist_ok=True)
os.makedirs(no_tumor_directory, exist_ok=True)

# Loop through the binary values in y_train and the corresponding image file names
for label, image_name in zip(y_test, image_filenames_valid):
    # Get the full path of the image file
    image_path = os.path.join(image_directory, image_name)
    
    if label == 1:
        # Move file to tumor directory
        shutil.move(image_path, os.path.join(tumor_directory, image_name))
        print(f'Moved {image_name} to Tumor directory')
    elif label==0:
        # Move file to no_tumor directory
        shutil.move(image_path, os.path.join(no_tumor_directory, image_name))
        print(f'Moved {image_name} to No Tumor directory')

print("File organization complete!")


# In[3]:


import torch,torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torchvision import models
from torch.optim import lr_scheduler


# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from tempfile import TemporaryFile


# In[5]:


from torchvision import datasets, transforms
import os


# In[8]:


dataset_path ='/Users/saibhargavmandava/Documents/bhushanGit/SDS-CP024-neurovision/submissions-team/bhushan/brain tumor/brain-tumor'
data_transforms ={
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ]),
    'valid': transforms.Compose([
       transforms.Resize(256),
         transforms.CenterCrop(224),    
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
image_datasets ={
    x: datasets.ImageFolder(os.path.join(dataset_path,x),data_transforms[x]) for x in ['train','valid']
}
image_loader ={
    x: torch.utils.data.DataLoader(image_datasets[x],batch_size=32,shuffle=True,num_workers=4) for x in ['train','valid']
}
data_sizes = {x: len(image_datasets[x]) for x in ['train','valid']}
class_names = image_datasets['train'].classes


# In[9]:


print(data_sizes['train'],data_sizes['valid'])
print(class_names)


# In[10]:


def imshow(inp,title=None):
    inp=inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
inputs, classes = next(iter(image_loader['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# In[11]:


device  = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f"using device {device}")


# In[19]:


from tempfile import TemporaryDirectory


# In[22]:


def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_acc=0
        for epoch in range(num_epochs):
            print(f"epoch {epoch}/{num_epochs-1}")  
            for phase in ['train','valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs,labels in image_loader[phase]:
                    inputs =inputs.to(device)
                    labels =labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase =='train'):
                        outputs=model(inputs)
                        _,preds=torch.max(outputs,1)
                        loss = criterion(outputs,labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        running_corrects += torch.sum(preds == labels.data)
                        running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / data_sizes[phase]
                epoch_acc = running_corrects.float() / data_sizes[phase]
                print(f"{phase} loss: {epoch_loss} acc: {epoch_acc}")
                if phase == 'train':
                    scheduler.step()
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    print(f"model saved to {best_model_params_path}")
            
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
        print(f'Best val acc: {best_acc}')
        model.load_state_dict(torch.load(best_model_params_path))       
    return model


# In[40]:


def visualize_model(model, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i,(inputs,labels) in enumerate(image_loader['valid']):
            inputs =inputs.to(device)
            labels =labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]} | actual: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# In[33]:


model_ft =models.resnet18(pretrained=True)
num_ftrs=model_ft.fc.in_features
model_ft.fc=nn.Linear(num_ftrs,2)
model_ft =model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001,momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)


# In[35]:


model_ft = train_model(model_ft,criterion,optimizer_ft,exp_lr_scheduler,num_epochs=25)


# In[41]:


visualize_model(model_ft)


# In[31]:


model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# In[32]:


model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)


# In[42]:


visualize_model(model_conv)

plt.ioff()
plt.show()


# In[45]:


get_ipython().system('pip install sklearn')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install seaborn')


# In[48]:


import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in image_loader['valid']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # Collect predictions
            all_labels.extend(labels.cpu().numpy())  # Collect actual labels

    # Compute Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# In[49]:


plot_confusion_matrix(model_ft)


# In[ ]:



