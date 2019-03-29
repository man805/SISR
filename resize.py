import argparse
import os
from PIL import Image


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'HR')):
        os.makedirs(os.path.join(output_dir, 'HR'))
    if not os.path.exists(os.path.join(output_dir, 'LR')):
        os.makedirs(os.path.join(output_dir, 'LR'))

    images = os.listdir(os.path.join(image_dir, 'HR'))
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, 'HR', image), 'r+b') as f:
            with Image.open(f) as img:
                hr_img = resize_image(img, size)
                hr_img.save(os.path.join(output_dir, 'HR', image), hr_img.format)
                lr_img = resize_image(img, [int(size[0] / 2), int(size[1] / 2)])
                lr_img.save(os.path.join(output_dir, 'LR', image), lr_img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

def main(args):
    train_image_dir = args.train_image_dir
    train_output_dir = args.train_output_dir
    #valid_image_dir = args.valid_image_dir
    #valid_output_dir = args.valid_output_dir
    image_size = [args.image_size, args.image_size]
    
    resize_images(train_image_dir, train_output_dir, image_size)
    #resize_images(valid_image_dir, valid_output_dir, image_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_dir', type=str, default='./data/mscoco2017_val',
                        help='directory for train images')
    parser.add_argument('--valid_image_dir', type=str, default='./data/val2014/',
                        help='directory for validation images')
    parser.add_argument('--train_output_dir', type=str, default='./data/mscoco2017_val_resized/',
                        help='directory for saving resized train images')
    parser.add_argument('--valid_output_dir', type=str, default='./data/valid_resized2014/',
                        help='directory for saving resized validation images')
    parser.add_argument('--image_size', type=int, default=128,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)