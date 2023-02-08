from PIL import Image, ImageFont, ImageDraw


def get_wrapped_text(text: str, font, line_length: int):
    lines = ['']
    text = text.replace('-', ' ').replace('_', ' ')
    for word in text.split():
        line = f'{lines[-1]} {word}'.strip()
        print('Checking if line is too long for line length: ', line, line_length)
        if font.getlength(line) <= line_length:
            lines[-1] = line
        else:
            lines.append(word)
    return '\n'.join(lines)


def get_fontsize(width, height, text_length):
    fontsize = 1  # starting font size
    font = ImageFont.truetype("/content/compare_diffusion/Monaco.ttf", fontsize)
    # portion of image width you want text width to be
    img_fraction = 0.95
    while font.getlength(text_length) < img_fraction * width and font.getsize(text_length)[1] < img_fraction * height:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("/content/compare_diffusion/Monaco.ttf", fontsize)
    return fontsize - 1


def create_text_image(text, width, height, x_justify=0.05, y_justify=0.5,
                      font=ImageFont.truetype("/content/compare_diffusion/Monaco.ttf", 43),
                      wrap=False):
    img = Image.new('RGB', (width, height), color=(255, 255, 255))

    draw = ImageDraw.Draw(img)

    if text == '' or text is None:
        return img
    elif wrap:
        text = get_wrapped_text(text, font, width)

    # Position text
    x = width * x_justify
    y = height * y_justify

    # Draw the text
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    return img
