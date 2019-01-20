# Imports here
import torch
import numpy as np
import pandas as pd
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
import os
import json



# Function that loads a checkpoint and rebuilds the model (VGG16)
def load_checkpoint(model_file):
    checkpoint = torch.load(model_file)
    
    model = models.vgg16(pretrained=True)
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    # model specifics
    model.class_to_idx = checkpoint['class_to_idx']
    #model.epochs = checkpoint['epochs']
    
    #optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # images have to be normalized
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    # preprocess step
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    loaded_image = Image.open(image)
    img_tensor = preprocess(loaded_image).float()
    img_tensor = img_tensor.unsqueeze(0)
    np_image = np.array(img_tensor)
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    # preprocess image
    img_tensor = process_image(image_path)
    img_tensor = torch.from_numpy(img_tensor)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        model = model.cuda()

    output = model(img_tensor)
    # softmax layer to convert the output to probabilities
    m = nn.Softmax()
    output = m(output)

    probabilities = torch.topk(output,5)[0]
    labels = torch.topk(output,5)[1]
    
    if torch.cuda.is_available():
        probabilities = probabilities.cpu()
        labels = labels.cpu()
    
    return img_tensor, probabilities.detach().numpy(), labels.detach().numpy()

def main():
    test_img_path = '/home/ec2-user/test_images/image_05188.jpg'
    class_values_json = 'cat_to_name.json'
    topk = 5
    model_file = 'model_flower_classifier.pt'
    
    #class_values_json = 'cat_to_name.json'
    #model_file = input("Input the path to the model file for prediction: ")
    #topk = int(input("How many possible predictions to be displayed per input? (topK): "))
    #test_img_path = input("Input the path to the image to be predicted: ")

    with open(class_values_json, 'r') as f:
        cat_to_name = json.load(f)
    
    model, optimizer = load_checkpoint(model_file)
    _, probs, preds = predict(test_img_path, model, topk)


    idx_to_name = {category: cat_to_name[str(category)] for category in preds[0]}
    classes = list(idx_to_name.values())

    print("Predicting what kind of flower this is...")
    print("...")
    print("...")
    print('This flower is most likely a {} with a {:.6f} probability!'.format(classes[0], probs[0][0]))

    print()
    print('This flower could also possibly be:')
    for i in range(len(probs[0])):
        if i != 0:
            print('{} with a {:.6f} probability'.format(classes[i], probs[0][i]))

main()
