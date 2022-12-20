import os

from PIL import Image

from pdf_generator import create_row_from_paths, create_header_row, create_subsection_header_row, vertical_concat_images


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

def load_files(paths, section_param, subsection_param, row_param, column_param):
    # Create an empty dictionary to store the sections and subsections
    sections = {}

    # Iterate through each path in the list
    for path in paths:
        # Split the path into a list of components using '/' as the delimiter
        components = path.split('/')

        # Extract the values for the section, subsection, row, and column parameters
        section = get_parameter_value(path, section_param)
        subsection = get_parameter_value(path, subsection_param)
        row = get_parameter_value(path, row_param)
        column = get_parameter_value(path, column_param)

        # If the section doesn't already exist in the dictionary, add it
        if section not in sections:
            sections[section] = {}

        # If the subsection doesn't already exist in the dictionary, add it
        if subsection not in sections[section]:
            sections[section][subsection] = {}

        # If the row doesn't already exist in the subsection, add it
        if row not in sections[section][subsection]:
            sections[section][subsection][row] = {}

        # Add the path to the appropriate row and column in the subsection
        sections[section][subsection][row][column] = path

    return sections


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
                keys.extend(extract_keys_from_nested_dict(value, n-1))
        return keys



image_paths = get_all_images_in_subtree('output')
files = load_files(image_paths, 'model', 'cfg', 'den', 'seed')

row_headers = sorted(list(set(extract_keys_from_nested_dict(files, 2))))
col_headers = sorted(list(set(extract_keys_from_nested_dict(files, 3))))
# Create a column header row
column_header_row = create_header_row(col_headers)

width = 512
height = 512
rows_per_page = 10
page_width = width * (len(col_headers) + 1)

for section in files:
    subsection_page_list = []
    for subsection in files[section]:
        # Create a subsection header
        subsection_header_row = create_subsection_header_row('Subsection Param: ' + subsection, page_width, int(height / 2))
        # Create page list
        page_rows = [subsection_header_row, column_header_row]
        rows = files[section][subsection]
        for row_header in row_headers:
            row = rows[row_header]
            row_paths = []
            for col_header in col_headers:
                row_paths.append(row[col_header])
            print(row_header)
            image_row = create_row_from_paths(row_paths, row_header)
            if len(page_rows) % rows_per_page == 0:
                page_rows.append(image_row)
                subsection_page = Image.fromarray(vertical_concat_images(page_rows))
                subsection_page_list.append(subsection_page)
                page_rows = [subsection_header_row, column_header_row]
            page_rows.append(image_row)
        if len(page_rows) > 2:
            subsection_page = Image.fromarray(vertical_concat_images(page_rows))
            subsection_page_list.append(subsection_page)



    for page in subsection_page_list:
        page.show()
    # Create a section page
    # section_page = create_section_page(section)

#
# horizontal_concat_paths = get_all_horizontal_concat_paths()