import os

import numpy as np
from PIL import Image, ImageFont, ImageDraw


def get_parameter_value(path, parameter):
    # Split the path by '/' to get a list of individual components
    path_components = path.split('/')

    # Find the component that starts with the parameter name followed by '_'
    for component in path_components:
        param_prefix = parameter + '_'
        if component.startswith(param_prefix):
            # Split the component by '_' and return the second element (the value)
            # Strip the curly braces from the value
            return component.replace(param_prefix, '').strip('{}')

    # If the parameter was not found, return None
    return None


def load_files(paths, row_param, column_param):
    # Create an empty dictionary to store the sections and subsections
    rows = {}

    # Iterate through each path in the list
    for path in paths:

        # Extract the values for row, and column parameters
        row = get_parameter_value(path, row_param)
        column = get_parameter_value(path, column_param)

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
    return create_text_image(subsection, width, height, x_justify=0)


def create_y_axis_title(title, width, height):
    return create_text_image(title, width, height)


def create_x_axis_title(title, width, height):
    image = create_text_image(title, height, width, x_justify=0.3).rotate(90, expand=True)
    return image


def create_header_row(row_headers, image_width=512, height=512):
    images = [create_text_image(row_header, image_width, height) for row_header in row_headers]
    images.insert(0, create_blank_image(int(image_width / 3), height))
    return horizontal_concat_PIL_images(images)


def create_row_from_paths(paths, row_header, width, height):
    images = get_PIL_images_from_paths(paths)
    images.insert(0, create_text_image(row_header, int(width / 3), height))
    pil_images = horizontal_concat_PIL_images(images)
    print('CREATED HORIZONTAL CONCAT WITH WIDTH: ', pil_images.width)
    return pil_images


def create_blank_image(width, height):
    return Image.new('RGB', (width, height), color=(255, 255, 255))


def create_text_image(text='final font size', width=512, height=512, x_justify=0.5, y_justify=0.5, fontsize=43):
    img = Image.new('RGB', (width, height), color=(255, 255, 255))

    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("/content/compare_diffusion/Monaco.ttf", fontsize)

    print(text, fontsize)

    # Position text
    x = width * x_justify
    y = height * y_justify

    # Draw the text
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

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


def generate_pdf(x_axis, y_axis, width, height, hidden_params, rows_per_page=10,
                 generated_images_path='output'):
    image_paths = get_all_images_in_subtree(generated_images_path)
    files = load_files(image_paths, y_axis, x_axis)
    row_headers = sorted(list(set(extract_keys_from_nested_dict(files, 0))))
    col_headers = sorted(list(set(extract_keys_from_nested_dict(files, 1))))
    # Create hidden params row
    # Create a column header row
    column_header_row = create_header_row(col_headers, image_width=width, height=int(height / 5))
    page_width = (width * (len(col_headers))) + int(width / 3)
    print('PAGE WIDTH:', page_width)
    page_list = []
    # Create a subsection header
    subsection_headers = []
    for hidden_param in hidden_params.keys():
        hidden_param_string = hidden_param + ': ' + str(hidden_params[hidden_param])
        subsection_header = create_subsection_header_row(hidden_param_string, page_width, int(height / 6))
        subsection_headers.append(subsection_header)
    # Create axis titles
    x_axis_title = create_y_axis_title(x_axis, page_width, int(height / 3))
    # Create page list
    page_rows = subsection_headers + [x_axis_title, column_header_row]
    num_header_rows = len(page_rows)
    for row_header in row_headers:
        row = files[row_header]
        row_paths = []
        for col_header in col_headers:
            row_paths.append(row[col_header])
        image_row = create_row_from_paths(row_paths, row_header, width, height)
        page_rows.append(image_row)
        if len(page_rows) % rows_per_page == 0:
            page = Image.fromarray(vertical_concat_images(page_rows))
            page_list.append(page)
            page_rows = subsection_headers + [x_axis_title, column_header_row]
    if len(page_rows) > num_header_rows:
        page = Image.fromarray(vertical_concat_images(page_rows))
        page_list.append(page)

    # Create pdf
    final_pages = []
    for idx, page in enumerate(page_list):
        # get page width and height
        page_width, page_height = page.size
        y_axis_title = create_x_axis_title(y_axis, int(width / 3), page_height)
        final_pages.append(Image.fromarray(horizontal_concat_images([y_axis_title, page])))

    if len(final_pages) > 1:
        final_pages[0].save('output.pdf', save_all=True, append_images=final_pages[1:], optimize=True)
    else:
        final_pages[0].save('output.pdf', optimize=True)