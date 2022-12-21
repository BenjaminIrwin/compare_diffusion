import argparse
import os

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, StableDiffusionInpaintPipeline

from pdf_gen import generate_pdf


def terminal_progress_bar(current, total, bar_length=20, task='Image Generation'):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent / 100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    print('%s Progress: [%s%s] %d %%' % (task, arrow, spaces, percent), end='\r')


def get_all_images_in_subtree(root_dir):
    images = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                images.append(os.path.join(root, file))
    return images


def arrange_images(setup, image_paths):
    rows_param_name = setup.get('rows')
    cols_param_name = setup.get('cols')
    subsection_param_name = setup.get('subsections')
    section_param_name = setup.get('sections')

    sections = group_image_paths_into_dict_by_param(image_paths, section_param_name)
    for section_key in sections.keys():
        subsections = group_image_paths_into_dict_by_param(sections[section_key], subsection_param_name)
        for subsection_key in subsections.keys():
            rows = group_image_paths_into_dict_by_param(subsections[subsection_key], rows_param_name)
            for row_key in rows.keys():
                cols = group_image_paths_into_dict_by_param(rows[row_key], cols_param_name)
                for col in cols:
                    # Order cols properly
                    pass
                rows[row_key] = cols
            subsections[subsection_key] = rows
        sections[section_key] = subsections


def group_image_paths_into_dict_by_param(image_paths, param_name):
    image_paths_by_param = {}
    for image_path in image_paths:
        param = image_path.split(param_name + '_')[1].split('/')[0]
        if param not in image_paths_by_param:
            image_paths_by_param[param] = []
        image_paths_by_param[param].append(image_path)
    return image_paths_by_param


def complete_sentences(strings):
    # Initialize an empty list to store the completed sentences
    sentences = []

    # Initialize an empty string to store the current sentence
    current_sentence = ""

    # Iterate through the strings
    for string in strings:
        # Add the string to the current sentence
        current_sentence += string + " "

        # If the current string ends with a quotation mark, it is the end of a sentence
        if string[-1] == "'":
            # Add the completed sentence to the list of sentences and reset the current sentence
            sentences.append(current_sentence)
            current_sentence = ""

        # Return the list of sentences
    return sentences


def get_image_paths(dir):
    image_paths = []
    for root, dirs, files in os.walk(dir):
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


def load_files(paths, section_param, subsection_param, row_param, column_param):
    # Create an empty dictionary to store the sections and subsections
    sections = {}

    # Iterate through each path in the list
    for path in paths:
        # Split the path into a list of components using '/' as the delimiter
        components = path.split('/')

        # Extract the values for the section, subsection, row, and column parameters
        section = components[components.index(section_param) + 1]
        subsection = components[components.index(subsection_param) + 1]
        row = components[components.index(row_param) + 1]
        column = components[components.index(column_param) + 1]

        # If the section doesn't already exist in the dictionary, add it
        if section not in sections:
            sections[section] = {}

        # If the subsection doesn't already exist in the dictionary, add it
        if subsection not in sections[section]:
            sections[section][subsection] = []

        # If the row doesn't already exist in the subsection, add it
        if len(sections[section][subsection]) < row:
            sections[section][subsection].append([])

        # Add the path to the appropriate row and column in the subsection
        sections[section][subsection][row][column] = path

    return sections

parser = argparse.ArgumentParser(description='Stable Diffusion Output Comparison')
parser.add_argument('--hf_token', type=str, required=True)
parser.add_argument('--type', type=str, required=True, choices=['img2img', 'txt2img', 'inpaint'])
parser.add_argument('--rows', type=str, required=True)
parser.add_argument('--cols', type=str, required=True)
parser.add_argument('--models', type=str, nargs='+', required=True)
parser.add_argument('--cfg_scale_list', type=float, nargs='+', required=True)
parser.add_argument('--denoising_strength_list', type=float, nargs='+', required=True)
parser.add_argument('--prompts', type=str, nargs='+', required=True)
parser.add_argument('--negative_prompts', type=str, nargs='*', default=[''])
parser.add_argument('--seeds', type=int, nargs='*', default=[1])
parser.add_argument('--height', type=int, default=512)
parser.add_argument('--width', type=int, default=512)

# /content/compare_diffusion/output4/dreamlike-diffusion-1.0/pmt_a cat eating food/neg_pmt_tongue/cfg_7.5/den_0.75/seed_2/i000000025.jpg
# /content/compare_diffusion/output4/dreamlike-diffusion-1.0/pmt_a cat eating food/neg_pmt_tongue/cfg_7.5/den_0.75/seed_2/i000000026.jpg

def generate_images():
    output_counter = 0

    for model_path in model_paths:
        output_folder_name = 'output4'
        if type == 'txt2img':
            model = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=hf_token,
                                                            torch_dtype=torch.float16)
        elif type == 'img2img':
            model = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, use_auth_token=hf_token,
                                                                   torch_dtype=torch.float16)
        elif type == 'inpaint':
            model = StableDiffusionInpaintPipeline.from_pretrained(model_path, use_auth_token=hf_token,
                                                                      torch_dtype=torch.float16)
        model = model.to("cuda")
        print('loaded_model: ' + model_path)
        for prompt in prompts:
            for negative_prompt in negative_prompts:
                for cfg_scale in cfg_scale_list:
                    for denoising_strength in denoising_strength_list:
                        for seed in seeds:
                            generator = torch.Generator("cuda").manual_seed(seed)
                            if type == 'txt2img':
                                folder = f'{output_folder_name}/model_{model_path.split("/")[-1]}/pmt_{prompt}/' \
                                         f'neg_pmt_{negative_prompt}/cfg_{cfg_scale}/seed_{seed}'
                                create_folder(folder)
                                try:
                                    # Call txt2img
                                    output = model(prompt=prompt, guidance_scale=cfg_scale, generator=generator,
                                                   negative_prompt=negative_prompt, height=height, width=width).images[
                                        0]
                                    # Generate image name as increment of previous image
                                    output.save(folder + '/output_' + str(output_counter) + '.png')
                                    output_counter += 1
                                    terminal_progress_bar(output_counter, num_images_to_generate)
                                except Exception as e:
                                    print('Error generating image with params: ' + str(prompt) + ' ' + str(
                                        negative_prompt)
                                          + ' ' + str(cfg_scale) + ' ' + str(denoising_strength))
                                    print(e)
                            else:
                                folder = f'model_{output_folder_name}/{model_path.split("/")[-1]}/pmt_{prompt}/' \
                                         f'neg_pmt_{negative_prompt}/cfg_{cfg_scale}/den_{denoising_strength}/seed_{seed}'
                                create_folder(folder)
                                for idx, image in enumerate(images):
                                    # try:
                                    pil_image = Image.open(image)
                                    image_name = image.split('/')[-1]
                                    if type == 'inpaint':
                                        # Call inpaint
                                        pil_mask = Image.open(masks[idx])
                                        output = model(prompt=prompt, image=pil_image, mask_image=pil_mask,
                                                       guidance_scale=cfg_scale,
                                                       generator=generator, strength=denoising_strength, height=height,
                                                       width=width).images[0]
                                        output.save(folder + '/' + str(image_name))
                                        output_counter += 1
                                        terminal_progress_bar(output_counter, num_images_to_generate)
                                    elif type == 'img2img':
                                        # Call img2img
                                        output = model(prompt=prompt, image=pil_image, guidance_scale=cfg_scale,
                                                       generator=generator, strength=denoising_strength, height=height,
                                                       width=width).images[0]
                                        output.save(folder + '/' + str(image_name))
                                        output_counter += 1
                                        terminal_progress_bar(output_counter, num_images_to_generate)
                                    # except Exception as e:
                                    #     print('Error generating image with params: ' + str(prompt) + ' ' + str(
                                    #         negative_prompt) + ' ' + str(cfg_scale) + ' ' + str(denoising_strength))
                                    #     print(e)


if __name__ == "__main__":
    args = parser.parse_args()
    type = args.type
    if type == 'img2img':
        # Filter out images that are not valid
        images, masks = [image_path for image_path in get_image_paths('input/images') if validate(image_path)], None
    elif type == 'inpaint':
        images, masks = clean(get_image_paths('input/images'), get_image_paths('input/masks'))
    else:
        images, masks = None, None

    height = args.height
    width = args.width
    model_paths = args.models
    cfg_scale_list = args.cfg_scale_list
    if type == 'txt2img':
        denoising_strength_list = [0.0]
    else:
        denoising_strength_list = args.denoising_strength_list
    prompts = args.prompts
    negative_prompts = args.negative_prompts
    seeds = args.seeds
    hf_token = args.hf_token

    num_images_to_generate = len(model_paths) * len(cfg_scale_list) * len(denoising_strength_list) * \
                             len(prompts) * len(negative_prompts) * len(seeds)
    if images:
        num_images_to_generate *= len(images)

    print('**** Input parameters ****')
    print('Type: ' + type)
    print(f'Number of images to generate: {num_images_to_generate}')
    print(f'Models: {model_paths}')
    print(f'Cfg_scale_list: {cfg_scale_list}')
    print(f'Denoising_strength_list: {denoising_strength_list}')
    print(f'Prompts: {prompts}')
    print(f'Negative prompts: {negative_prompts}')
    print(f'Seeds: {seeds}')
    if images is not None:
        print(f'Number of input images: {len(images)}')
        if masks is not None:
            print(f'Number of input masks: {len(masks)}')

    hidden_params = {}
    hidden_params['type'] = type
    hidden_params['height'] = height
    hidden_params['width'] = width
    if len(model_paths) == 1:
        hidden_params['model'] = model_paths[0]
    if len(cfg_scale_list) == 1:
        hidden_params['cfg_scale'] = cfg_scale_list[0]
    if len(denoising_strength_list) == 1:
        hidden_params['denoising_strength'] = denoising_strength_list[0]
    if len(prompts) == 1:
        hidden_params['prompt'] = prompts[0]
    if len(negative_prompts) == 1:
        hidden_params['negative_prompt'] = negative_prompts[0]
    if len(seeds) == 1:
        hidden_params['seed'] = seeds[0]

    generate_images()
    generate_pdf(args.rows, args.cols, width, height, hidden_params)
