import os

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

from mask import get_crop_region, expand_crop_region

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


def apply_overlay(base_image, paste_loc, overlay):
    if overlay is not None and paste_loc is not None:
        x, y, width, height = paste_loc
        overlay = resize_image(1, overlay, width, height)
        base_image.paste(overlay, (x, y))

    return base_image


def resize_image(resize_mode, im, width, height):
    def resize(im, w, h):
        return im.resize((w, h), resample=LANCZOS)

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                      box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                      box=(fill_width + src_w, 0))

    return res


def generate_images(args, images, masks):
    hf_token = args['hf_token']
    output_path = args['output_path']
    height = args['height']
    width = args['width']
    model_paths = args['model']
    prompts = args['prompt']
    cfg_scale_list = args['cfg_scale']
    denoising_strength_list = args['denoising_strength']
    negative_prompts = args['negative_prompt']
    seeds = args['seed']
    inference_type = args['type']
    inpaint_full_res = args['inpaint_full_res']
    inpaint_full_res_padding = args['inpaint_full_res_padding']
    steps_list = args['steps']

    output_counter = 0

    print('Generating images...')

    if not inference_type == 'txt2img' and len(images) == 0:
        raise ValueError('No images provided for img2img/inpainting')

    for model_path in model_paths:
        model = get_model(hf_token, model_path, inference_type).to("cuda")
        for prompt in prompts:
            for negative_prompt in negative_prompts:
                for cfg_scale in cfg_scale_list:
                    for denoising_strength in denoising_strength_list:
                        for steps in steps_list:
                            for seed in seeds:
                                generator = torch.Generator("cuda").manual_seed(seed)
                                if inference_type == 'txt2img':
                                    folder = f'{output_path}/mo_{model_path.split("/")[-1]}/pr_{prompt}/' \
                                             f'ne_{negative_prompt}/cf_{cfg_scale}/st_{steps}/se_{seed}'
                                    if not os.path.exists(folder):
                                        os.makedirs(folder)
                                    try:
                                        print('Generating image with params: ' + str(prompt) + ' ' + str(negative_prompt)
                                              + ' ' + str(cfg_scale) + ' ' + str(denoising_strength))
                                        # Call txt2img
                                        output = model(prompt=prompt, guidance_scale=cfg_scale, generator=generator,
                                                       negative_prompt=negative_prompt, height=height, width=width,
                                                       num_inference_steps=steps).images[0]
                                        # Generate image name as increment of previous image
                                        output.save(folder + '/output_' + str(output_counter) + '.png')
                                        output_counter += 1
                                    except Exception as e:
                                        print('Error generating image with params: ' + str(prompt) + ' ' + str(
                                            negative_prompt)
                                              + ' ' + str(cfg_scale) + ' ' + str(denoising_strength))
                                        print(e)
                                else:
                                    folder = f'{output_path}/mo_{model_path.split("/")[-1]}/pr_{prompt}/' \
                                             f'ne_{negative_prompt}/cf_{cfg_scale}/de_{denoising_strength}/st_{steps}/' \
                                             f'se_{seed}'
                                    if not os.path.exists(folder):
                                        os.makedirs(folder)
                                    for idx, image in enumerate(images):
                                        print(image)
                                        try:
                                            pil_image = Image.open(image)
                                            pil_mask = Image.open(masks[idx])
                                            image_name = 'im_' + image.split('/')[-1]
                                            if inference_type == 'inpaint':
                                                print('Inpainting image with params: ' + str(prompt) + ' ' + str(
                                                    negative_prompt)
                                                      + ' ' + str(cfg_scale) + ' ' + str(denoising_strength))
                                                if inpaint_full_res:
                                                    paste_to, pil_image, pil_mask = full_res_transform(
                                                        inpaint_full_res_padding,
                                                        pil_image,
                                                        pil_mask)

                                                output = model(prompt=prompt, negative_prompt=negative_prompt, image=pil_image.convert('RGB'),
                                                               mask_image=pil_mask.convert('RGB'),
                                                               guidance_scale=cfg_scale, generator=generator,
                                                               height=height, width=width, num_inference_steps=steps).images[0]
                                                if inpaint_full_res:
                                                    output = apply_overlay(Image.open(image), paste_to, output)
                                                output.save(folder + '/' + str(image_name))
                                            elif inference_type == 'img2img':
                                                print('Generating image with params: ' + str(prompt) + ' ' + str(
                                                    negative_prompt) + ' ' + str(cfg_scale) + ' ' + str(denoising_strength))
                                                output = model(prompt=prompt, negative_prompt=negative_prompt, image=pil_image, guidance_scale=cfg_scale,
                                                               generator=generator, strength=denoising_strength,
                                                               height=height, width=width,num_inference_steps=steps).images[0]
                                                output.save(folder + '/' + str(image_name))
                                        except Exception as e:
                                            print('Error generating image with params: ' + str(prompt) + ' ' + str(
                                                negative_prompt) + ' ' + str(cfg_scale) + ' ' + str(denoising_strength))
                                            print(e)


def full_res_transform(inpaint_full_res_padding, pil_image, pil_mask):
    monochannel_mask = pil_mask.convert('L')
    crop_region = get_crop_region(np.array(monochannel_mask),
                                  inpaint_full_res_padding)
    crop_region = expand_crop_region(crop_region, pil_image.width,
                                     pil_image.height, monochannel_mask.width,
                                     monochannel_mask.height)
    x1, y1, x2, y2 = crop_region
    monochannel_mask = monochannel_mask.crop(crop_region)
    cropped_image = pil_image.crop(crop_region)
    pil_mask = resize_image(2, monochannel_mask, pil_image.width,
                            pil_image.height)
    pil_image = resize_image(2, cropped_image, pil_image.width,
                             pil_image.height)
    paste_to = (x1, y1, x2 - x1, y2 - y1)
    return paste_to, pil_image, pil_mask


def get_model(hf_token, model_path, inference_type):
    if inference_type == 'txt2img':
        return StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=hf_token,
                                                       torch_dtype=torch.float16)
    elif inference_type == 'img2img':
        return StableDiffusionImg2ImgPipeline.from_pretrained(model_path, use_auth_token=hf_token,
                                                              torch_dtype=torch.float16)
    elif inference_type == 'inpaint':
        return StableDiffusionInpaintPipeline.from_pretrained(model_path, use_auth_token=hf_token,
                                                              torch_dtype=torch.float16)
    else:
        raise ValueError(f'Invalid type: {inference_type}')
