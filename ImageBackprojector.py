import argparse
import os
import sys
import pickle
from typing import List

import numpy as np

if "stylegan2" not in sys.path:
    sys.path.append("stylegan2")
import stylegan2.run_projector as proj
from cv2 import cv2
from Data import ck


# TODO Split this into another file and take the pickle as an argument
# TODO Partition out images so we can test with a smaller batch
def imageManip(network, images, run_pct: float = 1.):
    images = images[:int(run_pct * len(images))]
    latents_array = np.zeros((len(images), 18, 512), np.uint8)
    projector = proj.prep_project_image(network)

    for i, image in enumerate(images):
        # Resize to 256x256
        image = cv2.resize(image, (256, 256))
        # Go from HxWxC to CxHxW
        image = image.transpose(2, 0, 1)
        # Insert a batch size of 1
        image = np.expand_dims(image, axis=0)
        # project and append to list of latents
        latents_array[i] = proj.project_target(image, projector)[0]

        print(f"{(i+1)/len(images) * 100:03.2f}% Complete")

    # latents2 = [proj.project_image_nosave(sys.argv[1], np.expand_dims(i.transpose(2, 0, 1), axis=0)) for i in images]
    print("Latent Array Shape: ", latents_array.shape)
    return latents_array


def get_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network", required=True, help="StyleGAN2 Network pkl")
    parser.add_argument("-d", "--data", required=True, help="File to dump latent data to")
    return parser.parse_args(argv)

def main():
    args = get_args(sys.argv[1:])

    _, images, _, _ = ck.getLastFrameData((256, 256), True)
    data = imageManip(args.network, images)

    data_file = args.data if args.data.endswith(".pkl") else args.data + ".pkl"
    with open(data_file, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()

