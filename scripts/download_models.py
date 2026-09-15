import os
import urllib.request

MODELS = {
    "weights/hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "weights/mobile_sam.pt": "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"
}

def download_weights():
    os.makedirs("weights", exist_ok=True)
    for path, url in MODELS.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            urllib.request.urlretrieve(url, path)
            print(f"Downloaded {path} successfully.")
        else:
            print(f"{path} already exists. Skipping.")

if __name__ == "__main__":
    download_weights()