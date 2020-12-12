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
SORCERY using the command below. Our installation required additional linking
of the gcc/g++ dependency, commands for which can be seen below if your system
requries the same.

Environment Creation:
```sh
conda env create -f conda_environment.yml
```

GCC/G++ Linking:
```sh
ln -s envs/SORCERY/bin/x86_64-conda_cos6-linux-gnu-gcc envs/SORCERY/bin/gcc
ln -s envs/SORCERY/bin/x86_64-conda_cos6-linux-gnu-g++ envs/SORCERY/bin/g++
```

If you plan to run the StyleGAN2-Encoder, you will also need to install azure-cognitiveservices-vision-computervision, as seen below.
```sh
pip install azure-cognitiveservices-vision-computervision cognitive-face
```
## Running the project

### Data Loading

The data is loaded up automatically when the classifier is run. The user will have to specify the path for CK+.

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
The classifier is run in a two step process.
1. Get the latent spaces. This is done through the ImageBackprojector.py script
   by providing a network to use for latent spaces, and a desired filename
   prefix

Help Menu:
```python
usage: ImageBackprojector.py [-h] -n NETWORK -d DATA

optional arguments:
  -h, --help            show this help message and exit
  -n NETWORK, --network NETWORK
                        StyleGAN2 Network pkl
  -d DATA, --data DATA  File to dump latent data to
  -c CK, --ck CK        Path to CK Data
```

Example:
```sh
python ImageBackprojector.py -network [NETWORK_FILE].pkl -d "MyLatents" --ck ck_data/
```

2. Train the network. This is done through the RunModel.py script by providing
   the training and test data as generated by the ImageBackprojector.py script,
   a minimum percentage of presence in an action unit, a number of epochs, a
   batch size, an interval to halve the learning rate at, and if you would like
   to test the best network upon completion.

Help Menu:
```python
usage: RunModel.py [-h] --train_data TRAIN_DATA --test_data TEST_DATA
                   [-p PERCENT_ACTIVE] [-e EPOCHS] [-b BATCH_SIZE]
                   [-l LR_INTERVAL] [-t]

optional arguments:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA
                        Train data pkl from ImageBackprojector.py
  --test_data TEST_DATA
                        Test data pkl from ImageBackprojector.py
  -p PERCENT_ACTIVE, --percent_active PERCENT_ACTIVE
                        Minimum percentage of activation of AUs as a decimal
  -e EPOCHS, --epochs EPOCHS
                        Epochs to train for
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training
  -l LR_INTERVAL, --lr_interval LR_INTERVAL
                        Number of epochs between learning rate halving
  -t, --test            Test the model after training has completed
  -c CK, --ck CK        Path to CK Data
```

Example:
```sh
python RunModel.py --train_data data_new-training.pkl --test_data data_new-test.pkl --ck ck-data/ -e 500 -l 100 --test
```

### StyleGAN2-Encoder
1. The StyleGAN2-Encoder first uses NVIDIA's StyleGAN2 run_projector.py script to
get and store latent spaces of images. An example of this command can be found
below, but refer to the StyleGAN2 documentation for any additional information
```sh
python run_projector.py project-generated-images --network gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds 0,1,5 --result-dir backprojection/
```

2. Next we run the FaceLabelGenerator.py script. This script requires a directory
of images to run the annotation tool on, an Azure subscription key, an Azure
endpoint, and an optional output file name and recognition model. I will not be
providing an example to this as it requires a subscription key, but the
relevant arguments can be seen below.
```sh
usage: FaceLabelGenerator.py [-h] -i IMAGES -k KEY -e ENDPOINT [-d {1,2,3}]
                             [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGES, --images IMAGES
                        Image Directory
  -k KEY, --key KEY     Azure subscription key
  -e ENDPOINT, --endpoint ENDPOINT
                        Azure endpoint
  -d {1,2,3}, --recognition_model {1,2,3}
                        Model used for detection
  -o OUTPUT, --output OUTPUT
                        JSON file to write to
```

3. To learn latent spaces, we've provided a self-explanatory interactive notebook
at stylegan2/StyleGAN2-Encoder-LearnDirections.ipynb

4. To adjust facial attributes after learning latent spaces, we've provided a
notebook at stylegan2/StyleGAN2-Encoder-AdjustDirections.ipynb

