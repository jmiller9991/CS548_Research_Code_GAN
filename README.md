# Generating Facial Recognition Data using StyleGAN2

## Overview

This is the code repository for the paper “Generating Facial Recognition Data”
using StyleGAN2. This code was written as part of the CS 490/548 course taught
by Dr. Michael J. Reale at SUNY Polytechnic in Fall 2020. This code was written
by [Justin Firsching](https://github.com/JustinFirsching), [Andrew
Bertino](https://github.com/AndrewBert), [Jacob Miller
](https://github.com/jmiller9991), and [Dr. Michael J.
Reale](https://github.com/PrimarchOfTheSpaceWolves) . This is also based on the
projects [StyleGAN2](https://github.com/NVlabs/stylegan2) by Karras et al. and
[stylegan-encoder](https://github.com/Puzer/stylegan-encoder) by Puzer.

### Data Loading

The facial recognition training data is loaded from the CK+ database. 
CK+ contains 593 emotion sequences from 123 subjects. It contains four pieces of information
for each emotion sequence: images, facial action coding data, emotion details and landmarks. 
We have created an API to pull any of this information from the database. 

### Back-Projection

Back-projection is handled by the StyleGAN2 system. This system uses a
projector to produce the latent space. In the case of running the projector
directly, it will load a generator to aid in obtaining the latent space. Then
the projector will produce the latent space for an image but, as it works, it
produces a latent space every 100 steps. For the projection built in to
training the classifier, it does not save those intermediate steps.

### Classifier
The classifier is meant to take in the latent space representing an image of a
human face and obtain the action units present in the image. This system is not
complete, instead, it takes in the CK+ dataset from the data loader and
attempts to train using them.  It has a poor training capability and needs more
work to become a successful action unit classifier.
### StyleGAN2-Encoder

The StyleGAN2-Encoder takes advantage of Microsoft's Cognitive Services facial
analysis capabilities while leveraging the generative capabilities of
StyleGAN2. This is a simple three step process starting with obtaining the
facial annotations for a generated image. Next, these annotations are used to
train a linear model to learn how representations of an attribute change when
an attribute is present. Finally, these linear learnings of image
representations are connected to an interactive python interface allowing a
user to modify the "average" face, as understood by StyleGAN2, to make a facial
feature more or less present in the image.

## Dependencies and Setup
This project requires the use of [StyleGAN2](https://github.com/NVlabs/StyleGAN2)
with an ffhq pretrained model (we suggest
[this](https://drive.google.com/file/d/1igxv6ZP4TFGe_392B-qnSqXnglTKH5yo/view?usp=sharing)
one).

While Windows and Linux are supported, Linux is highly recommended for
compatibility. Additionally, at least one CUDA enabled GPU with CUDA version 10
and cuDNN 7.5 is required. This GPU must have at least 8GB of RAM.

We elected to use Anaconda for Python 3 as the base environment, using Python
3.7 with the Tensorflow 1.15, Jupyter, Keras, and Pillow libraries.

The project dependencies can be installed into a new Anaconda environment named
SORCERY using the command below.

```sh
conda env create -f conda_environment.yml
```

## Running the project

### Data Loading

### Back-Projection

The projector can be run two different ways, both of which can be seen below.
The first way is through the command line. If you want to project images that
have been generated, use the command:
```sh
python run_projector.py project-generated-images --network=<network-file> \
  --seeds=0,1,5
```
and if you want to project images that are real, use the command:
```sh
python run_projector.py project-real-images --network=<network-file> \
  --dataset=<type> --data-dir=~/datasets
```
In both cases, <network-file> is the file that the network is saved to as a pkl
file and the <type> value is the type of pictures like car for cars and ffhq
for faces. For more information, please see the StyleGAN2 documentation at the
StyleGAN2 repository.

In addition, you can run the training of the classifier in which
back-projection will occur and a file will be saved of the set of images used
for training.

### Classifier

### StyleGAN2-Encoder
