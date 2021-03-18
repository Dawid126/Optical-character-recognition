import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import cv2


pattern_chars_roboto = ["a", "d", "e", "f",  "g", "q", "h", "b", "4", "t", "p", "j", "s",
                 "k", "m", "n", "u", "r", "6", "o", "l", "w", "c", "x",
                 "y", "v", "z", "0", "8", "9", "1", "i", "2", "3",
                 "5", "7", "exclamation", "question", "comma", "dot"]

pattern_chars_sans = ["a", "d", "e", "f",  "g", "q", "b", "h","t", "4", "0", "j", "s",
                 "k", "m", "p", "n", "l", "u", "r", "6", "o", "w", "c", "x",
                 "y", "v", "z", "8", "9", "1", "i", "2", "3",
                 "5", "7", "exclamation", "question", "comma", "dot"]

special_chars = {"exclamation": "!", "question": "?", "dot": ".", "comma": ","}


char_coeffs_roboto = {"a":0.92, "b":0.9, "c":0.9, "d":0.9, "e":0.9, "f":0.9, "g":0.9,
               "h":0.9, "i":0.85, "j":0.91, "k":0.9, "l":0.85, "m":0.91,
               "n":0.9, "o":0.9, "p":0.9, "q":0.91, "r":0.89, "s":0.92,
               "t":0.9, "u":0.88, "v":0.87, "w":0.9, "x":0.9, "y":0.9,
               "z":0.9, "0": 0.96, "1": 0.97, "2":0.93, "3":0.93, "4":0.9,
               "5":0.9, "6":0.9, "7":0.9, "8":0.9, "9":0.94, "exclamation":0.87,
               "question":0.87, "comma":0.8, "dot":0.8}

char_coeffs_sans = {"a":0.9, "b":0.9, "c":0.9, "d":0.9, "e":0.9, "f":0.9, "g":0.9,
               "h":0.9, "i":0.89, "j":0.91, "k":0.9, "l":0.92, "m":0.90,
               "n":0.9, "o":0.9, "p":0.92, "q":0.91, "r":0.91, "s":0.89,
               "t":0.9, "u":0.88, "v":0.87, "w":0.9, "x":0.9, "y":0.9,
               "z":0.9, "0": 0.96, "1": 0.97, "2":0.93, "3":0.93, "4":0.9,
               "5":0.9, "6":0.9, "7":0.9, "8":0.92, "9":0.94, "exclamation":0.87,
               "question":0.87, "comma":0.8, "dot":0.8}

pattern_images = {}
chars_count = {}
chars_positions = []



def rotate_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    file_name = image_path.split(".")[0] + "-rotated.png"
    cv2.imwrite(file_name, rotated)

def save_image_as_chart_with_markes(array_image, markers_location, marker_shapes,
                                    file_name, title ="", x_label = "", y_label = ""):

    height, width = np.shape(array_image)
    fig, ax1 = plt.subplots(figsize=(width / 50, height / 50))
    ax1.imshow(array_image, cmap="gray")
    ax1.set(xlabel=x_label, ylabel=y_label, title=title)

    for i in range(np.shape(markers_location)[0]):
        for j in range(np.shape(markers_location)[1]):
            if markers_location[i, j] != 0:
                mark = ptch.Rectangle((j, i), - marker_shapes[1], - marker_shapes[0], linewidth=1,
                                      edgecolor="r", facecolor="none")
                ax1.add_patch(mark)

    fig.savefig(f"step_by_step/{file_name}.png")


def load_patterns(patterns_path, pattern_chars, pattern_images):
    for char in pattern_chars:
        pattern_path = patterns_path + f"{char}.png"
        pattern_images[char] = Image.open(pattern_path).convert("L")
        chars_count[char] = 0


def load_original_text(text_path):
    txt_file = open(text_path, "r")
    text = txt_file.read()
    txt_file.close()
    return text


def compute_correlation(image_array, pattern_array, coeff):
    pattern_array_rotated = np.rot90(pattern_array, 2)
    pattern_fft = np.fft.fft2(pattern_array_rotated, np.shape(image_array))

    image_fft = np.fft.fft2(image_array)

    correlation_array = np.real(np.fft.ifft2(np.multiply(pattern_fft, image_fft)))
    max = np.max(correlation_array)
    for i in range(np.shape(correlation_array)[0]):
        for j in range(np.shape(correlation_array)[1]):
            if(correlation_array[i,j] < max * coeff):
                correlation_array[i, j] = 0

    return correlation_array


def add_chars(image_array, correlation_array, char, pattern_shapes, chars_positions, chars_count):
    for i in range(np.shape(correlation_array)[0]):
        for j in range(np.shape(correlation_array)[1]):
            if(correlation_array[i,j] != 0):
                chars_positions.append((i, j, char))
                chars_count[char] = chars_count[char] + 1

                correlation_array[i - pattern_shapes[0] //2 : i + pattern_shapes[0] // 2,
                j - pattern_shapes[1] //2 : j + pattern_shapes[1] // 2] = 0

                image_array[i - pattern_shapes[0] : i + 1 ,
                j - pattern_shapes[1] : j + 1 ] = 0


def match_all_chars(image_array, pattern_images, chars_positions, chars_count, char_coeffs, pattern_chars, step_by_step=False):
    counter = 0
    for char in pattern_chars:

        pattern_image = ImageOps.invert(pattern_images[char])
        pattern_array = np.asarray(pattern_image)

        correlation_array = compute_correlation(image_array, pattern_array, char_coeffs[char])
        if(step_by_step):
            counter = counter + 1
            save_image_as_chart_with_markes(image_array, correlation_array, np.shape(pattern_image), counter)

        add_chars(image_array, correlation_array, char, np.shape(pattern_image), chars_positions, chars_count)


def print_count(char, original_text, chars_count):
    found_count = chars_count[char]
    if(char in special_chars):
        original_count = original_text.lower().count(special_chars[char])
    else:
        original_count = original_text.lower().count(char)
    print(f"{char}\t\t{found_count}\t\t{original_count}\t\t{found_count / original_count * 100} %")


def show_found_to_original_ratio(pattern_chars, chars_count, original_text):
    print("Char\tFound\tOriginal\tRatio ")
    for char in pattern_chars:
        if(char in original_text.lower()):
            print_count(char, original_text, chars_count)
        elif(char in special_chars and special_chars[char] in original_text):
            print_count(char, original_text, chars_count)


def convert_to_text(chars_positions, pattern_images, chars_count, special_chars):
    lines = []
    line_height = 15
    space_width = 5
    text = ""
    line = []
    chars_positions.sort(key=lambda x:x[0])

    for i in range(len(chars_positions) - 1):
        line.append((chars_positions[i][1], chars_positions[i][2]))
        if(chars_positions[i+1][0] - chars_positions[i][0] >= line_height):
            lines.append(line)
            line = []

    line.append((chars_positions[-1][1], chars_positions[-1][2]))
    lines.append(line)

    second_char = None

    for line in lines:
        line.sort(key=lambda x: x[0])
        for i in range(len(line) - 1):
            first_char_pos = line[i][0]
            second_char_pos = line[i+1][0]
            first_char = line[i][1]
            second_char = line[i+1][1]
            second_char_width = np.shape(pattern_images[second_char])[1]

            if (first_char in special_chars):
                first_char = special_chars[first_char]
            if (second_char in special_chars):
                second_char = special_chars[second_char]

            if(second_char_pos - first_char_pos >= space_width + second_char_width):
                text = text + first_char + " "
            else:
                text = text + first_char

        text = text + second_char + "\n"

    return text


def OCR(image_path, text_path, patterns_path, patern_images, chars_count, chars_positions,
        char_coeffs, pattern_chars, special_chars, step_by_step = True, show_ratio = True):

    load_patterns(patterns_path, pattern_chars, pattern_images)
    rotate_image(image_path)
    original_text = (load_original_text(text_path))
    rotated_image_path = image_path.split(".")[0] + "-rotated.png"
    image = Image.open(rotated_image_path).convert('L')
    image = ImageOps.invert(image)
    image_array = cv2.fastNlMeansDenoising(np.asarray(image))

    match_all_chars(image_array, pattern_images, chars_positions, chars_count, char_coeffs,
                    pattern_chars, step_by_step)
    print(convert_to_text(chars_positions, pattern_images, chars_count, special_chars))
    if(show_ratio):
        show_found_to_original_ratio(pattern_chars, chars_count, original_text)


OCR("resources/open-sans/open-sans-image-with-text-skewed.png", "resources/open-sans/open-sans-text-skewed.txt", "resources/open-sans/", pattern_images,
    chars_count, chars_positions ,char_coeffs_sans, pattern_chars_sans, special_chars, step_by_step = True, show_ratio = True)
