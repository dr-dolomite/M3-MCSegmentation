import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from model import UNET
from utils import load_checkpoint

# Load the pre-trained model checkpoint
checkpoint_path = "my_checkpoint.pth.tar"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNET(in_channels=3, out_channels=1)
load_checkpoint(torch.load(checkpoint_path), model)
model = model.to(device).eval()

# Load and preprocess the input image
image_path = "PUT_TEST_IMG/test.jpg"
image = Image.open(image_path).convert("RGB")
image = TF.resize(image, (400, 400), antialias=True)
image_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

# Perform segmentation on the input image
with torch.no_grad():
    prediction = model(image_tensor)
    mask = torch.sigmoid(prediction) > 0.5
    mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)

# Create the colored segmentation mask
color_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)  # Modified line
color_mask[mask[0] == 1] = (138, 43, 226)  # Violet color (RGB values)  # Modified line

# Convert the colored mask to PIL Image and save it
color_mask_image = Image.fromarray(color_mask)
color_mask_image.save("SEGMENTED_IMG/segmentation_output.png")