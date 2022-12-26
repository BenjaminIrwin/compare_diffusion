import argparse
import os
import warnings

from PIL import Image

from img_gen import generate_images
from pdf_gen import generate_pdf


def get_image_paths(parent_path):
    image_paths = []
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    return image_paths


def validate(fp: str) -> bool:
    try:
        Image.open(fp)
        return True
    except:
        print(f'WARNING: Image cannot be opened: {fp}')
        return False


def extract_input_num(path):
    return int(path.split('_')[1].split('.')[0])


def clean(image_files, mask_files):
    print('ImageStore: Sorting images and masks')

    clean_images = []
    clean_masks = []

    # Create a dictionary that maps image numbers to filenames
    image_dict = {}
    for img in image_files:
        # If image is valid
        if validate(img):
            # Extract the number from the filename
            img_num = extract_input_num(img)
            image_dict[img_num] = img

    # Create a dictionary that maps mask numbers to filenames
    mask_dict = {}
    for msk in mask_files:
        # If mask is valid
        if validate(msk):
            # Extract the number from the filename
            msk_num = extract_input_num(msk)
            mask_dict[msk_num] = msk

    # Iterate over the keys (numbers) in the image dictionary
    for img_num in image_dict:
        # Check if the number exists in the mask dictionary
        if img_num in mask_dict:
            # If it does, add the corresponding filenames to the clean lists
            clean_images.append(image_dict[img_num])
            clean_masks.append(mask_dict[img_num])

    return clean_images, clean_masks


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


output_folder_name = 'output5'
dim_choices = ['model', 'cfg_scale', 'denoising_strength', 'prompt', 'negative_prompt', 'seed']

parser = argparse.ArgumentParser(description='Compare Diffusion')
parser.add_argument('--hf_token', type=str, required=True)
parser.add_argument('--type', type=str, required=True, choices=['img2img', 'txt2img', 'inpaint'])
parser.add_argument('--rows', type=str, required=True, choices=dim_choices)
parser.add_argument('--cols', type=str, required=True, choices=dim_choices)
parser.add_argument('--models', type=str, nargs='+', required=True)
parser.add_argument('--cfg_scale_list', type=float, nargs='+', required=True)
parser.add_argument('--denoising_strength_list', type=float, nargs='+', required=True)
parser.add_argument('--prompts', type=str, nargs='+', required=True)
parser.add_argument('--negative_prompts', type=str, nargs='*', default=[''])
parser.add_argument('--seeds', type=int, nargs='*', default=[1])
parser.add_argument('--height', type=int, default=512)
parser.add_argument('--width', type=int, default=512)


def get_hidden_params():
    hidden_params = {'type': (args['type']), 'height': (args['height']), 'width': (args['width'])}
    for dim in dim_choices:
        if args['rows'] != dim and args['cols'] != dim:
            if len(args['dim']) > 1:
                warnings.warn(f"More than one '{dim}' path provided, but rows nor cols are set to '{dim}'. Defaulting "
                              f"to '{dim}' at index 0.")
            hidden_params[dim] = args['dim'][0]

    return hidden_params


if __name__ == "__main__":
    args = parser.parse_args().__dict__
    images, masks = None, None
    if args['type'] == 'txt2img':
        args['denoising_strength_list'] = [0.0]
    else:
        if args['type'] == 'img2img':
            images = [image_path for image_path in get_image_paths('input/images') if validate(image_path)]
        elif args['type'] == 'inpaint':
            images, masks = clean(get_image_paths('input/images'), get_image_paths('input/masks'))

    generate_images(args['hf_token'], args['output_path'], args['height'], args['width'], images, masks,
                    args['model_paths'], args['prompts'], args['cfg_scale_list'], args['denoising_strength_list'],
                    args['negative_prompts'], args['seeds'])
    hidden_params = get_hidden_params()
    generate_pdf(args['cols'], args['rows'], args['width'], args['height'], hidden_params,
                 generated_images_path=output_folder_name)
