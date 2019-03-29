import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import os
import torch.nn as nn
from torchvision import transforms
from data_loader import get_loader
from model import SISR
from PIL import Image
from math import log10


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([transforms.ToTensor()])
    criterion = nn.MSELoss(reduction='mean')

    model = SISR()
    # Build models
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    valid_data_loader = get_loader(args.valid_image_dir, transform, 1, shuffle=True, num_workers=3)

    avg_PSNR = 0
    with torch.no_grad():
        for i, (hr_image, lr_image) in enumerate(valid_data_loader):
            # Set mini-batch dataset
            hr_image = hr_image.to(device)
            lr_image = lr_image.to(device)
            
            # Forward, backward and optimize
            outputs = model(lr_image)
            loss = criterion(outputs, hr_image)

            # Print log info
            avg_PSNR += 10 * log10(1 / loss.item())

    print('Validation PSNR: %f' % (float(avg_PSNR) / 100))

    fig = plt.figure()
    # Prepare an image
    for i, index in enumerate([15, 69, 96]):
        lr_image = load_image('./data/mscoco2017_val_resized/LR/%04d.jpg' % index, transform)
        lr_image_tensor = lr_image.to(device)
        
        # Generate an caption from the image
        with torch.no_grad():
            outputs = model(lr_image_tensor)
        hr_image = Image.open('./data/mscoco2017_val_resized/HR/%04d.jpg' % index)
        
        # print(outputs.size())
        # print(torch.squeeze(outputs).size())
        # print(torch.squeeze(outputs).permute(1, 2, 0).size())
        # exit()
        plt.subplot(231+i)
        plt.imshow(np.asarray(hr_image))
        # plt.imshow(np.asarray(hr_image))
        plt.subplot(234+i)
        image = (torch.squeeze(outputs).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        plt.imshow(image)
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path for trained decoder')
    parser.add_argument('--valid_image_dir', type=str, default='data/mscoco2017_val_resized', help='directory for resized validation images')
    # Model parameters (should be same as paramters in train.py)
    args = parser.parse_args()
    main(args)