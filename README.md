# Generating Facial Recognition Data using StyleGAN2

***Overview***

This is the code repository for the paper “Generating Facial Recognition Data” using StyleGAN2. This code was written as part of the CS 490/548 course taught by Dr. Michael J. Reale at SUNY Polytechnic in Fall 2020. This code was written by Justin Firsching ([JustinFirsching](https://github.com/JustinFirsching) and [justin-firsching](https://github.com/justin-firsching) on GitHub), Andrew Bertino ([AndrewBert](https://github.com/AndrewBert) and [TheCrackSpider](https://github.com/TheCrackSpider) on GitHub), Jacob Miller ([jmiller9991](https://github.com/jmiller9991) on GitHub), and Dr. Michael J. Reale ([PrimarchOfTheSpaceWolves
](https://github.com/PrimarchOfTheSpaceWolves) on GitHub). This is also based on the projects [StyleGAN2](https://github.com/NVlabs/stylegan2) by Karras et al. and [stylegan-encoder](https://github.com/Puzer/stylegan-encoder) by Puzer.

*Data Loading*

*Back-Projection*

Back-projection is handled by the StyleGAN2 system. This system uses a projector to produce the latent space. In the case of running the projector directly, it will load a generator to aid in obtaining the latent space. Then the projector will produce the latent space for an image but, as it works, it produces a latent space every 100 steps. For the projection built in to training the classifier, it does not save those intermediate steps. 

*Classifier*

*StyleGAN2-encoder*

***Dependencies and Setup***

***Running the project***

*Data Loading*

*Back-Projection*

In order to use the projector, it can be run in two ways. The first way is through the command line. If you want to project images that have been generated, use the command:
```python run_projector.py project-generated-images --network=gdrive:<network-file> \
  --seeds=0,1,5
```
and if you want to project images that are real, use the command:
```python run_projector.py project-real-images --network=gdrive:<network-file> \
  --dataset=<type> --data-dir=~/datasets
```
In both cases, <network-file> is the file that the network is saved to as a pkl file and the <type> value is the type of pictures like car for cars and ffhq for faces. For more information, please see the StyleGAN2 documentation at the StyleGAN2 repository.

In addition, you can run the training of the classifier in which back-projection will occur and a file will be saved of the set of images used for training.

*Classifier*

*StyleGAN2-encoder*
