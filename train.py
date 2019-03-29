import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from data_loader import get_loader
from torchvision import transforms
from model import SISR
from tensorboardX import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    basic_model_path = './models/'
    if not os.path.exists(basic_model_path + args.model_name):
        os.makedirs(basic_model_path + args.model_name)
    basic_log_path = './logs/'
    if not os.path.exists(basic_log_path + args.model_name):
        os.makedirs(basic_log_path + args.model_name)
    logger = SummaryWriter(basic_log_path + args.model_name)

    '''
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    # Build data loader
    train_data_loader = get_loader(args.train_image_dir, transform, args.batch_size, shuffle=True, num_workers=args.num_workers) 
    valid_data_loader = get_loader(args.valid_image_dir, transform, 1, shuffle=True, num_workers=args.num_workers)
    print('train_data_loader length : %d' % len(train_data_loader))
    print('valid_data_loader length : %d' % len(valid_data_loader))

    # Build the models
    model = SISR().to(device)
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='mean')
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(train_data_loader)
    for epoch in range(args.num_epochs):
        avg_train_loss = 0
        avg_valid_loss = 0

        for i, (hr_image, lr_image) in enumerate(train_data_loader):
            # Set mini-batch dataset
            hr_image = hr_image.to(device)
            lr_image = lr_image.to(device)
            
            # Forward, backward and optimize
            outputs = model(lr_image)
            loss = criterion(outputs, hr_image)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            avg_train_loss += loss.item() * hr_image.shape[0]
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item() / float(hr_image.shape[0])))     

        # Save the model checkpoints   
        torch.save(model.state_dict(), os.path.join(
            basic_model_path + args.model_name, 'SISR-{}.ckpt'.format(epoch+1)))
        logger.add_scalar('Train Loss', float(avg_train_loss) / 4900, epoch+1)

        model.eval()

        with torch.no_grad():
            for i, (hr_image, lr_image) in enumerate(valid_data_loader):
                # Set mini-batch dataset
                hr_image = hr_image.to(device)
                lr_image = lr_image.to(device)
                
                # Forward, backward and optimize
                outputs = model(lr_image)
                loss = criterion(outputs, hr_image)

                # Print log info
                avg_valid_loss += loss.item()

        logger.add_scalar('Valid Loss', float(avg_valid_loss) / 100, epoch+1)
        
        model.train()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('GPU version')
    else:
        print('CPU version')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--train_image_dir', type=str, default='data/mscoco2017_train_resized', help='directory for resized train images')
    parser.add_argument('--valid_image_dir', type=str, default='data/mscoco2017_val_resized', help='directory for resized validation images')
    parser.add_argument('--log_step', type=int , default=100, help='step size for prining log info')
    
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    if args.model_name == None:
        print('Please Run with model_name parameter')
        exit()
    
    print(args)
    main(args)