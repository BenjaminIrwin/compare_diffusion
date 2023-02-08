import os

import numpy as np
from PIL import Image, ImageFont

from text_gen import get_wrapped_text, create_text_image


def get_parameter_from_path(path, param_prefix):
    # Split the path by '/' to get a list of individual components
    path_components = path.split('/')

    # Find the component that starts with the parameter name followed by '_'
    for component in path_components:
        if component.startswith(param_prefix):
            # Split the component by '_' and return the second element (the value)
            return component.replace(param_prefix, '')

    # If the parameter was not found, return None
    return None


def load_files(paths, row_param, col_param):
    # Get first two letters f row and col params
    row_param_prefix = row_param[:2] + '_'
    col_param_prefix = col_param[:2] + '_'

    # Create an empty dictionary to store the sections and subsections
    rows = {}

    # Iterate through each path in the list
    for path in paths:

        # Extract the values for row, and column parameters
        row = get_parameter_from_path(path, row_param_prefix)
        column = get_parameter_from_path(path, col_param_prefix)

        # If the section doesn't already exist in the dictionary, add it
        if row not in rows:
            rows[row] = {}

        # Add the path to the appropriate row and column in the subsection
        rows[row][column] = path

    return rows


def get_all_images_in_subtree(root_dir):
    images = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                images.append(os.path.join(root, file))
    return images


def extract_keys_from_nested_dict(d, n):
    if n == 0:
        return d.keys()
    else:
        keys = []
        for key, value in d.items():
            if isinstance(value, dict):
                keys.extend(extract_keys_from_nested_dict(value, n - 1))
        return keys


def horizontal_concat_PIL_images(images):
    return Image.fromarray(horizontal_concat_images([np.array(image) for image in images]))


def horizontal_concat_images(images):
    return np.concatenate(images, axis=1)


def vertical_concat_PIL_images(images):
    return Image.fromarray(vertical_concat_images([np.array(image) for image in images]))


def vertical_concat_images(images):
    return np.concatenate(images, axis=0)


def create_hidden_param_row(hidden_params_string, width, height):
    height = get_col_header_height(hidden_params_string, width, height,
                                   font=ImageFont.truetype("/content/compare_diffusion/Monaco.ttf", 50))
    return create_text_image(hidden_params_string, width, height, x_justify=0.01, y_justify=0.01, wrap=True)


def create_cols_title(title, width, height):
    return create_text_image(title, width, height, x_justify=0.6, y_justify=0.5,
                             font=ImageFont.truetype("/content/compare_diffusion/Monaco.ttf", 50))


def create_rows_title(title, width, height):
    return create_text_image(title, height, width, x_justify=0.3, y_justify=0.5,
                             font=ImageFont.truetype("/content/compare_diffusion/Monaco.ttf", 50)).rotate(90,
                                                                                                          expand=True)


def create_cols_axis(col_headers, width, height, font, indent_width):
    images = [create_text_image(col_header, width, height, font=font,
                                wrap=True, x_justify=0, y_justify=0) for col_header in col_headers]
    images.insert(0, create_blank_image(indent_width, height))
    return horizontal_concat_PIL_images(images)


def create_row_from_paths(paths, row_header, width, height, font):
    # image_width, height, fontsize, wrap
    images = get_PIL_images_from_paths(paths)
    text_image = create_text_image(row_header, width, height,
                                   font=font, wrap=True, y_justify=0.01)
    images.insert(0, text_image)
    pil_images = horizontal_concat_PIL_images(images)
    return pil_images


def create_blank_image(width, height):
    return Image.new('RGB', (width, height), color=(255, 255, 255))


def get_PIL_images_from_paths(image_paths):
    return [Image.open(path).convert('RGB') for path in image_paths]


def filter_paths_by_names(paths, names):
    return [path for path in paths if any(name in path for name in names)]


def get_row_header_width(longest_row_header, image_height, image_width, font):
    longest_row_header = longest_row_header.replace('-', ' ').replace('_', ' ')
    longest_word = max(longest_row_header.split(), key=len)
    longest_word_width = font.getsize(longest_word)[0]
    scale = 0.1
    found_width = False
    width = int(image_width * scale)
    while not found_width:
        wrapped_text = get_wrapped_text(longest_row_header, font, width)
        num_lines = wrapped_text.count('\n') + 1
        height = num_lines * font.getsize(longest_row_header)[1]
        if height < image_height and longest_word_width < width:
            found_width = True
        else:
            scale += 0.1
            width = int(image_width * scale)
    return width


def get_col_header_height(longest_col_header, image_width, image_height, font):
    scale = 0.1
    found_height = False
    height = int(image_height * scale)
    print('Image height: ', image_height)
    while not found_height:
        num_lines = get_wrapped_text(longest_col_header, font, image_width).count('\n') + 1
        height = num_lines * font.getsize(longest_col_header)[1]
        if height < image_height:
            found_height = True
        else:
            scale += 0.1
            height = int(image_height * scale)
            print('Increasing height to ', height)

    height += int(height * 0.05)
    return height


def generate_pdf(col_param, row_param, width, height, hidden_params, rows_per_page=10,
                 generated_images_path='output'):
    font = ImageFont.truetype("/content/compare_diffusion/Monaco.ttf", 43)
    image_paths = get_all_images_in_subtree(generated_images_path)
    print('Image paths: ', image_paths)
    files = load_files(image_paths, row_param, col_param)
    print('Files: ', files)
    print('Preparing headers...')
    col_headers, row_headers = get_headers(files)
    col_header_height = get_col_header_height(str(max(col_headers, key=len)), width, height, font)
    row_header_width = get_row_header_width(str(max(row_headers, key=len)), height, width, font)
    column_header_row = create_cols_axis(col_headers, width, col_header_height, font, row_header_width)
    cols_title = create_cols_title(col_param, column_header_row.width, int(height / 3))
    page_list = []
    page_rows = [cols_title, column_header_row]
    num_header_rows = len(page_rows)
    print('Preparing images...')
    for row_header in row_headers:
        row = files[row_header]
        row_paths = []
        for col_header in col_headers:
            row_paths.append(row[col_header])
        image_row = create_row_from_paths(row_paths, row_header, row_header_width, height, font)
        page_rows.append(image_row)
        if len(page_rows) % rows_per_page == 0:
            page = Image.fromarray(vertical_concat_images(page_rows))
            page_list.append(page)
            page_rows = [cols_title, column_header_row]
    if len(page_rows) > num_header_rows:
        page = Image.fromarray(vertical_concat_images(page_rows))
        page_list.append(page)
    final_pages = []
    for idx, page in enumerate(page_list):
        page_width, page_height = page.size
        rows_title = create_rows_title(row_param, int(width / 3), page_height)
        page = Image.fromarray(horizontal_concat_images([rows_title, page]))
        page_width = page.width
        hidden_param_string = get_hidden_params_string(hidden_params)
        hidden_param_row = create_hidden_param_row(hidden_param_string, page_width, height)
        page = vertical_concat_images([hidden_param_row, page])
        final_pages.append(Image.fromarray(page))
    print('Saving pdf...')
    if len(final_pages) > 1:
        final_pages[0].save('output.pdf', save_all=True, append_images=final_pages[1:], optimize=True)
    else:
        final_pages[0].save('output.pdf', optimize=True)


def get_headers(files):
    row_headers = sorted(list(set(extract_keys_from_nested_dict(files, 0))))
    col_headers = sorted(list(set(extract_keys_from_nested_dict(files, 1))))
    return col_headers, row_headers


def get_hidden_params_string(hidden_params):
    hidden_param_string = 'Hidden Params:: \n'
    for hidden_param in hidden_params.keys():
        hidden_param_string += hidden_param + ': ' + str(hidden_params[hidden_param]) + ', '
    hidden_param_string = hidden_param_string[:-2]
    return hidden_param_string
