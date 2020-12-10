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

*Classifier*

*StyleGAN2-encoder*
