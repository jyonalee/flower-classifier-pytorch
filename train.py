# Imports here
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib.pyplot as plt  
from glob import glob
from tqdm import tqdm
from PIL import Image
import os
import time
import copy
import json



def train_model(model, criterion, optimizer, gpu_mode, data_loader, n_epochs=10):
    # keeping track of best weights
    best_model_wts = copy.deepcopy(model.state_dict())
    valid_loss_min = np.Inf # track change in validation loss

    for epoch in range(1, n_epochs+1):
        # keep track of duration of each epoch
        since = time.time()
        running_losses  = {}
        for phase in ['train', 'valid']:
            # keep track of training and validation loss
            running_loss = 0.0
            valid_acc = 0.0
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            for data, target in data_loader[phase]:
                # move tensors to GPU if CUDA is available
                if gpu_mode:
                    data, target = data.cuda(), target.cuda()
                else:
                    data, target = data, target
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = model(data)
                    # calculate the batch loss
                    loss = criterion(output, target)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # perform a single optimization step (parameter update)
                        optimizer.step()
                        
                    # get prediction and accuracy in validate phase
                    else:
                        # convert output probabilities to predicted class
                        _, preds = torch.max(output, 1)    
                        valid_acc += torch.sum(preds == target.data)
                    
                    # update average running loss 
                    running_loss += loss.item()*data.size(0)

            # calculate average losses
            running_losses[phase] = running_loss/len(data_loader[phase].dataset)
        valid_accuracy = float(valid_acc)/float(len(data_loader['valid'].dataset))

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f} ({}/{})'.format(
            epoch, running_losses['train'], running_losses['valid'], valid_accuracy, valid_acc,len(data_loader['valid'].dataset)))

        # save model if validation loss has decreased
        if running_losses['valid'] <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min, running_losses['valid']))
            best_model_wts = copy.deepcopy(model.state_dict())
            valid_loss_min = running_losses['valid']
        
        elapsed_time = time.time() - since
        print()
        print("Epoch completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        torch.cuda.empty_cache()
        
    model.load_state_dict(best_model_wts)
    return model

# Save the checkpoint, for VGG16
def save_checkpoint(model, optimizer, model_file):
    model.class_to_idx = train_dataset.class_to_idx
    parameters = {
        'class_to_idx': model.class_to_idx,
        #'epochs': model.epochs,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer': optimizer,
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    torch.save(parameters, model_file)

def main():
    model_input = 'vgg16'
    learning_rate = 0.01
    hidden_units = 512
    training_epochs = 20
    data_dir = 'flower_data'    
    model_name = 'model_flower_classifier512'

    #model_input = input("Type in the desired model architecture to train the model (options: vgg16): ")
    #data_dir = input("Type in the directory of the training dataset: ")
    #learning_rate = float(input("Type in the desired learning rate to train the model (ie. 0.01): "))
    #hidden_units = int(input("Type in the desired number of hidden units to train the model (greater than 102): "))
    #training_epochs = int(input("Type in the desired iterations to train the model: "))
    #model_name = input("Name of the trained model: ")


    print('------------------------------------------')
    print('All ready! Now preparing to train model...')
    print('...')
    print()

    # check if using GPU
    gpu_mode = torch.cuda.is_available()
    print('Does this machine have a GPU and can Pytorch use it?')
    print(gpu_mode)

    print()
    print('Loading data and pre-processing...')
    print('...')
    print()

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    # Define batch size
    batch_size = 20
    # Define transforms for the training and validation sets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    validate_data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(
        train_dir,
        train_data_transforms)

    validate_dataset = datasets.ImageFolder(
        valid_dir,
        validate_data_transforms)



    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    data_loader = {}
    data_loader['train'] = train_loader
    data_loader['valid'] = validate_loader

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    print()
    print('Preparing to train model!')
    print('Building network...')
    print('...')
    print()

    if model_input == 'vgg16':
        # Build and train network
        model = models.vgg16(pretrained=True)

    # Freeze training for all layers
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[-1].in_features
    model.classifier[6] = nn.Sequential(
                          nn.Linear(num_features, 512), 
                          nn.ReLU(), 
                          nn.Dropout(0.4),
                          nn.Linear(512, len(cat_to_name)),
                          nn.LogSoftmax(dim=1))

    print(model)
    print('Generated Model has been printed!')
    print()
    print()


    # define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())


    if gpu_mode:
        model.cuda()
    else:
        model.cpu() 


    print('Commencing training...')
    print('...')
    print()
    model = train_model(model, criterion, optimizer, gpu_mode, data_loader, n_epochs=training_epochs)

    print('Saving best model...')
    save_checkpoint(model, optimizer, model_name + '.pt')
    print('Training Complete! BYE!')

main()
