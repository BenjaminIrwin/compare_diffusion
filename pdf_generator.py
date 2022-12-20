import numpy as np
from PIL import Image, ImageFont, ImageDraw

def horizontal_concat_images_from_paths(image_paths):
    return horizontal_concat_images([np.array(image) for image in get_PIL_images_from_paths(image_paths)])

def horizontal_concat_PIL_images(images):
    return Image.fromarray(horizontal_concat_images([np.array(image) for image in images]))


def horizontal_concat_images(images):
    return np.concatenate(images, axis=1)


def vertical_concat_PIL_images(images):
    return Image.fromarray(vertical_concat_images([np.array(image) for image in images]))


def vertical_concat_images(images):
    return np.concatenate(images, axis=0)

def create_subsection_header_row(subsection, width, height):
    return create_text_image(subsection, width, height)

def create_header_row(row_headers, width=512, height=512):
    images = [create_text_image(row_header, width, height) for row_header in row_headers]
    images.insert(0, create_blank_image(width, height))
    return horizontal_concat_PIL_images(images)

def create_row_from_paths(paths, row_header):
    images = get_PIL_images_from_paths(paths)
    images.insert(0, create_text_image(row_header))
    return horizontal_concat_PIL_images(images)

def create_blank_image(width, height):
    return Image.new('RGB', (width, height), color=(255, 255, 255))

def create_text_image(text='final font size', width=512, height=512):
    img = Image.new('RGB', (width, height), color=(255, 255, 255))

    draw = ImageDraw.Draw(img)

    # Choose a font
    fontsize = 1  # starting font size

    # portion of image width you want text width to be
    img_fraction = 0.6

    font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", fontsize)
    while font.getsize(text)[0] < img_fraction * width and font.getsize(text)[1] < img_fraction * height:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", fontsize)

    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1
    font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", fontsize)

    print(text, fontsize)

    # Draw the text
    draw.text((0, height / 3), text, fill=(0, 0, 0), font=font)

    return img


def get_PIL_images_from_paths(image_paths):
    return [Image.open(path).convert('RGB') for path in image_paths]


def split_list_into_chunks(list, chunk_max_size):
    return [list[i:i + chunk_max_size] for i in range(0, len(list), chunk_max_size)]


def filter_paths_by_names(paths, names):
    return [path for path in paths if any(name in path for name in names)]

def check_if_folder_exists(folder):
    import os
    return os.path.exists(folder)
