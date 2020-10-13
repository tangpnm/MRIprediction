import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
#MONAI
import monai
from monai.networks.nets import densenet121
from monai.transforms import \
    Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg

# load model
num_class = 4
torch.cuda.empty_cache()
device = torch.device("cpu")
model = densenet121(
    spatial_dims=2,
    in_channels=1,
    out_channels=num_class
).to(device)

PATH = 'best_metric_model.pth'
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

# convert image -> tensor
def transforms_image(image_bytes):

    image_transforms = Compose([
        # LoadPNG(image_only=True),
        ScaleIntensity(),
        ToTensor()
    ])

    print("t2")
    # print(type(image))
    input = image_transforms(image_bytes)
    input_test = input.float()
    print("t3")
    input_predict = input_test.unsqueeze(0).unsqueeze(0)

    return input_predict

# predict
def get_prediction(image_tensor):

    val_outputs = model(image_tensor.to(device)).argmax(dim=1)

    return val_outputs
