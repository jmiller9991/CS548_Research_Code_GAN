import argparse
import json
import os
import sys
import time

from glob import glob
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face import FaceClient, FaceClientConfiguration
from azure.cognitiveservices.vision.face.models import FaceAttributeType


ALL_FACE_ATTRS = [e.value for e in FaceAttributeType]


def get_args(argv=sys.argv[1:]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", required=True, help="Image Directory")
    parser.add_argument("-k", "--key", required=True, help="Azure subscription key")
    parser.add_argument("-e", "--endpoint", required=True, help="Azure endpoint")
    parser.add_argument("-d", "--recognition_model", type=int, choices=[1, 2, 3], default=3, help="Model used for detection")
    parser.add_argument("-o", "--output", default="faces.json", help="JSON file to write to")

    namespace = parser.parse_args(argv)
    namespace.recognition_model = f"recognition_0{namespace.recognition_model}"
    return namespace


def run_detection(image_path: str, recognition_model: str, face_client: FaceClient):
    image = open(image_path, "rb")
    face_data = face_client.face.detect_with_stream(image, True, True, ALL_FACE_ATTRS, recognition_model=recognition_model, raw=True).response.json()
    image.close()
    time.sleep(4)  # Sleep for just over 1/20th of a minute to avoid having to pay

    # Store the file path of the face
    face_data[0]["localFacePath"] = image_path
    return face_data[0]


def main():
    args = get_args()
    images = glob(os.path.join(args.images, "**", "*.png"), recursive=True)
    creds = CognitiveServicesCredentials(args.key)
    client = FaceClient(args.endpoint, creds)
    faces = []
    for i, image in enumerate(images):
        print(f"{i + 1}/{len(images)}")
        face = run_detection(image, args.recognition_model, client)
        faces.append(face)
    faces.sort(key=lambda d: d["localFacePath"])
    with open(args.output, "w") as f:
        json.dump(faces, f, indent=4)


if __name__ == "__main__":
    main()

