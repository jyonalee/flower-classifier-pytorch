# flower-classifier-pytorch
Classifying 102 different flower species from images with CNN (using pytorch)

## Project Overview
This project is the final assignment from the [Udacity PyTorch Scholarship Challenge from Facebook](https://www.udacity.com/facebook-pytorch-scholarship). 
The project focuses on training a convolutional neural network to classify 102 different flower 
species from a set of 6000 images ([dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)). Because of the small dataset, I have utilized pre-trained models 
such as VGG19 and ResNet152 and did transfer learning on top of these models to classify the flower 
images with an accuracy of about 93% from the test dataset.

## Prerequisite

[Python3](https://www.python.org/downloads/)

[Jupyter notebook](https://jupyter.org/install)

[PyTorch](https://pytorch.org/get-started/locally/)


## Usage
**_Image Classifier Project.ipynb_**    

Jupyter notebook which shows the entire process from data loading 
to exploration, training, testing and validation.


**_train.py_**                          

trains the model to classify flower species. Takes in hyperparameters as well as base model,


**_predict.py_**                        

predicts input image(s) of the top k (user input) likely flower species with corresponding 
probabilities. Takes in user input of path of trained model and path of test image to be predicted.

### Using [Google Colab](https://colab.research.google.com/)
Colab is a cloud hosted jupyter notebook provided to the public for free by Google. The cool thing with 
Colab is that you don't need to manage your own servers and you can also utilize the GPU for free!
Importing notebooks from Github is pretty easy. Just type in the below in your browser
```
https://colab.research.google.com/github/jyonalee/flower-classifier-pytorch/blob/master/Image Classifier Project.ipynb
```
This will load the notebook in colab. Since Colab provisions a new environment each time
access the service, you will need to set up the necessary environment and datasets to
the provisioned instance.
Run the following commands before executing anything on the notebook
```
# this snippet downloads the repo and dataset and sets it up for google colab useage
!git clone https://github.com/jyonalee/flower-classifier-pytorch.git
!mv flower-classifier-pytorch/* .
!wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip flower_data.zip
```
