import os

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_images(args, images, masks):
    hf_token = args['hf_token']
    output_path = args['output_path']
    height = args['height']
    width = args['width']
    model_paths = args['models']
    prompts = args['prompts']
    cfg_scale_list = args['cfg_scale_list']
    denoising_strength_list = args['denoising_strength_list']
    negative_prompts = args['negative_prompts']
    seeds = args['seeds']
    type = args['type']

    output_counter = 0

    for model_path in model_paths:
        model = get_model(hf_token, model_path, type).to("cuda")
        print('Loaded model: ' + model_path.split("/")[-1])
        for prompt in prompts:
            for negative_prompt in negative_prompts:
                for cfg_scale in cfg_scale_list:
                    for denoising_strength in denoising_strength_list:
                        for seed in seeds:
                            generator = torch.Generator("cuda").manual_seed(seed)
                            if type == 'txt2img':
                                folder = f'{output_path}/m_{model_path.split("/")[-1]}/p_{prompt}/' \
                                         f'n_{negative_prompt}/c_{cfg_scale}/s_{seed}'
                                create_folder(folder)
                                try:
                                    # Call txt2img
                                    output = model(prompt=prompt, guidance_scale=cfg_scale, generator=generator,
                                                   negative_prompt=negative_prompt, height=height, width=width).images[
                                        0]
                                    # Generate image name as increment of previous image
                                    output.save(folder + '/output_' + str(output_counter) + '.png')
                                    output_counter += 1
                                except Exception as e:
                                    print('Error generating image with params: ' + str(prompt) + ' ' + str(
                                        negative_prompt)
                                          + ' ' + str(cfg_scale) + ' ' + str(denoising_strength))
                                    print(e)
                            else:
                                folder = f'{output_path}/m_{model_path.split("/")[-1]}/p_{prompt}/' \
                                         f'n_{negative_prompt}/c_{cfg_scale}/d_{denoising_strength}/s_{seed}'
                                create_folder(folder)
                                for idx, image in enumerate(images):
                                    try:
                                        pil_image = Image.open(image)
                                        image_name = image.split('/')[-1]
                                        if type == 'inpaint':
                                            pil_mask = Image.open(masks[idx])
                                            output = model(prompt=prompt, image=pil_image.convert('RGB'),
                                                           mask_image=pil_mask.convert('RGB'),
                                                           guidance_scale=cfg_scale,
                                                           generator=generator, height=height, width=width).images[0]
                                            output.save(folder + '/' + str(image_name))
                                        elif type == 'img2img':
                                            output = model(prompt=prompt, image=pil_image, guidance_scale=cfg_scale,
                                                           generator=generator, strength=denoising_strength,
                                                           height=height, width=width).images[0]
                                            output.save(folder + '/' + str(image_name))
                                    except Exception as e:
                                        print('Error generating image with params: ' + str(prompt) + ' ' + str(
                                            negative_prompt) + ' ' + str(cfg_scale) + ' ' + str(denoising_strength))
                                        print(e)


def get_model(hf_token, model_path, type):
    if type == 'txt2img':
        return StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=hf_token,
                                                       torch_dtype=torch.float16)
    elif type == 'img2img':
        return StableDiffusionImg2ImgPipeline.from_pretrained(model_path, use_auth_token=hf_token,
                                                              torch_dtype=torch.float16)
    elif type == 'inpaint':
        return StableDiffusionInpaintPipeline.from_pretrained(model_path, use_auth_token=hf_token,
                                                              torch_dtype=torch.float16)
    else:
        raise ValueError(f'Invalid type: {type}')
