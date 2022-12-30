import os

import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

from mask import get_crop_region, expand_crop_region

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


def apply_overlay(image, paste_loc, overlay):
    if overlay is None:
        return image

    if paste_loc is not None:
        x, y, width, height = paste_loc
        base_image = Image.new('RGBA', (overlay.width, overlay.height))
        ratio = width / height
        src_ratio = image.width / image.height
        src_w = width if ratio > src_ratio else image.width * height // image.height
        src_h = height if ratio <= src_ratio else image.height * width // image.width
        resized = resize(image, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        base_image.paste(image, (x, y))
        image = base_image

    image = image.convert('RGBA')
    image.alpha_composite(overlay)
    image = image.convert('RGB')

    return image


def resize(im, w, h):
    return im.resize((w, h), resample=LANCZOS)


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

    output_counter = 0

    for model_path in model_paths:
        model = get_model(hf_token, model_path, inference_type).to("cuda")
        for prompt in prompts:
            for negative_prompt in negative_prompts:
                for cfg_scale in cfg_scale_list:
                    for denoising_strength in denoising_strength_list:
                        for seed in seeds:
                            generator = torch.Generator("cuda").manual_seed(seed)
                            if inference_type == 'txt2img':
                                folder = f'{output_path}/m_{model_path.split("/")[-1]}/p_{prompt}/' \
                                         f'n_{negative_prompt}/c_{cfg_scale}/s_{seed}'
                                if not os.path.exists(folder):
                                    os.makedirs(folder)
                                try:
                                    # Call txt2img
                                    output = model(prompt=prompt, guidance_scale=cfg_scale, generator=generator,
                                                negative_prompt=negative_prompt, height=height, width=width).images[0]
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
                                if not os.path.exists(folder):
                                    os.makedirs(folder)
                                for idx, image in enumerate(images):
                                    # try:
                                    pil_image = Image.open(image)
                                    pil_mask = Image.open(masks[idx])
                                    image_name = image.split('/')[-1]
                                    if inference_type == 'inpaint':
                                        if inpaint_full_res:
                                            print('INPAINTING FULL RES')
                                            mask = pil_mask.convert('L')
                                            image_masked = Image.new('RGBA', (pil_image.width, pil_image.height))
                                            image_masked.paste(pil_image.convert("RGBA"),
                                                               mask=ImageOps.invert(
                                                                   mask))
                                            crop_region = get_crop_region(np.array(mask),
                                                                          inpaint_full_res_padding)
                                            crop_region = expand_crop_region(crop_region, width,
                                                                             height, mask.width,
                                                                             mask.height)
                                            x1, y1, x2, y2 = crop_region

                                            mask = mask.crop(crop_region)
                                            pil_mask = images.resize_image(2, mask, width, height)
                                            paste_to = (x1, y1, x2 - x1, y2 - y1)
                                        else:
                                            paste_to = None
                                            image_masked = None

                                        output = model(prompt=prompt, image=pil_image.convert('RGB'),
                                                       mask_image=pil_mask.convert('RGB'),
                                                       guidance_scale=cfg_scale,
                                                       generator=generator, height=height, width=width).images[0]
                                        output = apply_overlay(output, paste_to, image_masked)
                                        output.save(folder + '/' + str(image_name))
                                    elif inference_type == 'img2img':
                                        output = model(prompt=prompt, image=pil_image, guidance_scale=cfg_scale,
                                                       generator=generator, strength=denoising_strength,
                                                       height=height, width=width).images[0]
                                        output.save(folder + '/' + str(image_name))
                                    # except Exception as e:
                                    #     print('Error generating image with params: ' + str(prompt) + ' ' + str(
                                    #         negative_prompt) + ' ' + str(cfg_scale) + ' ' + str(denoising_strength))
                                    #     print(e)


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
