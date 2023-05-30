# M3 Segmentation Model

<img src="pytorch_logo.jpg" alt="PyTorch Logo" width="600" height="200" style="object-fit: contain;">


An early version of our M3 Segmentation Model using U-Net Architecture. This repository contains the code for training and applying the segmentation model to images.

### Description

This project implements a segmentation model based on the U-Net architecture using PyTorch. The U-Net model is trained on a dataset to perform image segmentation, specifically for segmenting objects in medical images. The model has been designed to accurately delineate and identify specific structures in the images.

### Features

- U-Net architecture for image segmentation
- Training code with customizable hyperparameters
- Inference code to apply segmentation to new images
- Pre-trained checkpoint for quick start

### Pre-installation

Pytorch was not included in the requirements.txt. I recommend to create a conda virtual environment first and install Pytorch based on the generated pip command from its own documentation. 

### Pytorch Documentation
```
https://pytorch.org/get-started/locally/
```

You can also refer to this. However, it might get deprecated if changes in the documentation will be applied. Use the link documentation instead if these installation commands won't work.


### Using GPU - Only use if you have NVIDIA Graphics Card that supports CUDA.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Using CPU
```
pip3 install torch torchvision torchaudio

```


### Download Model here:
```
https://wvsueduph-my.sharepoint.com/:u:/g/personal/russel_yasol_wvsu_edu_ph/Efgjv6YteO5HuQubdhCAvgoBVUHpLwutoo8M5qN-0GcMBw?e=6Sf9Jd
```

Copy and paste inside the M3Segmentation folder.


## Installation

1. Clone the repository:

```
git clone https://github.com/dr-dolomite/M3Segmentation.git
```

2. Change into the project directory:
```
cd m3-segmentation-model
```

3. Create / Activate your virtual environment
```
conda create -n M3segment python=3.10 -y
conda activate M3segment
```

4. Install the required packages:
```
pip install -r requirements.txt
```

5. Perform train or test.


