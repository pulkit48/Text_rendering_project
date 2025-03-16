import cv2
import numpy as np
import random
import webcolors
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datasets import load_dataset
import re
import os
import zipfile
import time
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor

# import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# from transformers import BitsAndBytesConfig
# from transformers import pipeline

######
'''You may have to change the path here'''
######

zip_file_path = 'images.zip'
extract_to_path = 'images'
os.makedirs(extract_to_path, exist_ok=True)
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)



os.makedirs('final_dataset',exist_ok=True)
os.makedirs('final_dataset/win',exist_ok=True)
os.makedirs('final_dataset/lose1',exist_ok=True)
os.makedirs('final_dataset/lose2',exist_ok=True)
os.makedirs('final_dataset/lose3',exist_ok=True)


# Number of samples that we want to generate
# Currently it should be less than 1000 because we have 1000 sample only from the dataset
num_samples=1000

############## You can change these all functions.....

def rand_color():
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black',
              'gray', 'orange', 'purple', 'pink', 'brown', 'lime', 'navy']
    return random.choice(colors)

def create_plain_image():
    text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color = (255, 255, 255)  # White background
    plain_image = np.full((512, 512, 3), color, dtype=np.uint8)
    return plain_image, color, text_color

def get_random_font():
    fonts = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX
]
    selected_font = random.choice(fonts)
    if selected_font == cv2.FONT_ITALIC:
        selected_font = random.choice(fonts[:-1]) | cv2.FONT_ITALIC
    return selected_font

def shuffle_chars(text):
    text_list = list(text)
    random.shuffle(text_list)
    return "".join(text_list)

def drop_chars(text):
    text_list = list(text)
    number_of_chars = random.randint(1, max(1, len(text) - 1))
    indices = random.sample(range(len(text)), number_of_chars)
    for i in sorted(indices, reverse=True):
        del text_list[i]
    return "".join(text_list)

def repeat_chars(text):
    repeat_fraction = random.uniform(0, 0.6)
    text_list = list(text)
    num_to_repeat = int(len(text) * repeat_fraction)
    for _ in range(num_to_repeat):
        idx = random.randint(0, len(text_list) - 1)
        rpt_time=random.randint(1,5)
        while rpt_time>0:
            text_list.insert(idx, text_list[idx])
            rpt_time-=1
        # text_list.insert(idx, text_list[idx])
    return "".join(text_list)

def scramble_words(text):
    words = text.split()
    random.shuffle(words)
    return " ".join(words)

def replace_with_random_chars(text):
    char_list = (
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    )
    text_list = list(text)
    indices = random.sample(range(len(text_list)), random.randint(0, len(text_list) - 1))
    for ind in indices:
        text_list[ind] = random.choice(char_list)
    return "".join(text_list)

def wrap_text_iteratively(text, font, font_scale, font_thickness, image_width, max_attempts=10):
    for _ in range(max_attempts):
        max_width = max(50, min(image_width - 50, random.randint(image_width // 2, image_width - 50)))
        lines = []
        current_line = ""

        words = text.split()
        for word in words:
            test_line = f"{current_line} {word}".strip()
            text_size = cv2.getTextSize(test_line, font, font_scale, font_thickness)[0]
            if text_size[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        total_height = len(lines) * (cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1] + 10)
        if total_height < image_width:
            return lines
    raise ValueError("Text is too large to fit even after attempts.")

def is_contrast_sufficient(bg_color, text_color, threshold=4.5):
    """
    Check if the contrast ratio between the background and text is sufficient.

    Args:
        bg_color (tuple): Background color (R, G, B).
        text_color (tuple): Text color (R, G, B).
        threshold (float): Minimum contrast ratio to consider sufficient.

    Returns:
        bool: True if the contrast is sufficient, False otherwise.
    """
    def luminance(color):
        r, g, b = [c / 255.0 for c in color]
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    l1 = luminance(bg_color) + 0.05
    l2 = luminance(text_color) + 0.05
    contrast_ratio = max(l1, l2) / min(l1, l2)
    return contrast_ratio >= threshold


def generate_random_contrasting_color(bg_color, max_attempts=20):
    """
    Generate a random text color that contrasts with the background color.

    Args:
        bg_color (tuple): Background color (R, G, B).
        max_attempts (int): Maximum attempts to find a contrasting color.

    Returns:
        tuple: A random contrasting color (R, G, B).
    """
    for _ in range(max_attempts):
        random_color = tuple(random.randint(0, 255) for _ in range(3))
        if is_contrast_sufficient(bg_color, random_color):
            return random_color
    # Fallback to black or white if no suitable color is found
    return (255, 255, 255) if sum(bg_color) / 3 < 128 else (0, 0, 0)


def paste_multiline_text(text, margin=20,input_image=None):
    
    image, _, text_color = create_plain_image()
    if input_image is not None:
      
        # start=time.time()
        image = cv2.imread(input_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
        # print(time.time()-start)
    img_h, img_w = image.shape[:2]
    pixels = np.array(image)
    mean_color = tuple(np.mean(pixels, axis=(0, 1)).astype(int))
    text_color = generate_random_contrasting_color(mean_color)

    font = get_random_font()
    font_scale = random.uniform(1.0,1.5)
    # font_thickness = random.randint(1, 2)
    font_thickness = 1
    try:
        lines = wrap_text_iteratively(text, font, font_scale, font_thickness, img_w - 2 * margin)
    except ValueError as e:
        print(e)
        return None

    line_height = cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1] + 10
    total_text_height = len(lines) * line_height
    start_y = random.randint(margin, max(margin, img_h - total_text_height - margin))

    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        start_x = random.randint(margin, max(margin, img_w - text_size[0] - margin))
        y = start_y + i * line_height
        cv2.putText(image, line, (start_x, y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

    
    return Image.fromarray(image)


def apply_distortions(text, distortion_list, max_attempts=10):
    num_choices = random.randint(1, 5)
    distortion_methods = random.sample(distortion_list, num_choices)
    modified_text = text
    attempts = max_attempts

    while text == modified_text and attempts > 0:
        for method in distortion_methods:
            modified_text = method(text)
        attempts -= 1
    if len(modified_text)==0 or text==modified_text:
        num_choices=1
        return (text[int(0.2*len(text)):int(0.7*len(text))],num_choices)

    return (modified_text,num_choices)

def data_generation1(type,prompt_list,external_df=None,index=0):
  prompt_list1 = []
  image1_list, image2_list,image3_list,image4_list = [], [],[],[]
  st = set()
  distortion_list = [replace_with_random_chars, scramble_words, repeat_chars, drop_chars, shuffle_chars]

  with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

    for i in range(num_samples):
        if i%100==0:
            print(i)
        text = random.choice(prompt_list)
        full_text=f'''An image with white background with text "{text}".'''
        if(type==0):
            input_image=None
            image1= paste_multiline_text(text, margin=20,input_image=input_image)
        elif type==1:
            ind=random.choice(range(len(external_df)))

            input_image=external_df['image'][ind]['path']
            # start=time.time()
            image1= paste_multiline_text(text, margin=20,input_image=input_image)
            # print(time.time()-start)
            full_text=f'''An image with text "{text}" on background as {external_df['prompt'][ind]}'''


        # Apply distortions and generate images with corresponding num_choices
        text2,num_choices2=apply_distortions(text, distortion_list)
        image2 = paste_multiline_text(text2, margin=20, input_image=input_image)
        text3,num_choices3=apply_distortions(text, distortion_list)
        image3= paste_multiline_text(text3, margin=20, input_image=input_image)
        text4,num_choices4=apply_distortions(text, distortion_list)
        image4 = paste_multiline_text(text4, margin=20, input_image=input_image)

        # Create a list of images and their num_choices
        image_choices = [(image2, num_choices2), (image3, num_choices3), (image4, num_choices4)]

        # Sort images by num_choices (ascending order: least → most)
        image_choices.sort(key=lambda x: x[1])  # Sorting based on num_choices

        # Assign paths based on sorted order
        if image1 and all(img[0] for img in image_choices):  # Ensure all images exist
            win_path = f"final_dataset/win/{i+index}.png"
            lose1_path = f"final_dataset/lose1/{i+index}.png"  # Least num_choices
            lose2_path = f"final_dataset/lose2/{i+index}.png"  # Moderate num_choices
            lose3_path = f"final_dataset/lose3/{i+index}.png"  # Highest num_choices

            start=time.time()
              
            def save_image(img, path):
                # img.save(path)
                cv2.imwrite(path, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.submit(save_image, image1, win_path)
                executor.submit(save_image, image_choices[0][0], lose1_path)
                executor.submit(save_image, image_choices[1][0], lose2_path)
                executor.submit(save_image, image_choices[2][0], lose3_path)
            # print(time.time()-start)

            if i%50==0:
                pd.DataFrame([[full_text, win_path, lose1_path, lose2_path, lose3_path]], columns=columns)\
                    .to_csv(output_csv, mode='a', header=False, index=False)

def data_generation2(type,prompt_list,external_df=None,index=0):

  prompt_list1 = []
  image1_list, image2_list,image3_list,image4_list = [], [],[],[]
  st = set()
  distortion_list = [replace_with_random_chars, scramble_words, repeat_chars, drop_chars, shuffle_chars]


  for i in range(num_samples):
    if i%100==0:
        print(i)
    full_text=prompt_list[i]
    match = re.search(r"'(.*?)'", full_text)
    if match is None:
      continue
    text = match.group(1)

    if(type==2):
      input_image=None
      image1= Image.open(f'images/images/img{i}.jpg').convert("RGB")
      image1 = image1.resize((512, 512))
      # image1=np.array(image1)
    elif type==3:
      ind=random.choice(range(len(external_df)))
      input_image=external_df['image'][ind]['path']
      # full_text=f'''An image with text "{text}" on background as {external_df['prompt'][ind]}'''
      image1= Image.open(f'images/images/img{i}.jpg').convert("RGB")
      image1 = image1.resize((512, 512))
      # image1=np.array(image1)


    # Apply distortions and generate images with corresponding num_choices
    text2,num_choices2=apply_distortions(text, distortion_list)
    image2 = paste_multiline_text(text2, margin=20, input_image=input_image)
    text3,num_choices3=apply_distortions(text, distortion_list)
    image3= paste_multiline_text(text3, margin=20, input_image=input_image)
    text4,num_choices4=apply_distortions(text, distortion_list)
    image4 = paste_multiline_text(text4, margin=20, input_image=input_image)

    # Create a list of images and their num_choices
    image_choices = [(image2, num_choices2), (image3, num_choices3), (image4, num_choices4)]

    # Sort images by num_choices (ascending order: least → most)
    image_choices.sort(key=lambda x: x[1])  # Sorting based on num_choices

    # Assign paths based on sorted order
    if image1 and all(img[0] for img in image_choices):  # Ensure all images exist
        win_path = f"final_dataset/win/{i+index}.png"
        lose1_path = f"final_dataset/lose1/{i+index}.png"  # Least num_choices
        lose2_path = f"final_dataset/lose2/{i+index}.png"  # Moderate num_choices
        lose3_path = f"final_dataset/lose3/{i+index}.png"  # Highest num_choices

        # Save images to corresponding paths
        image1.save(win_path)  # Winner image
        image_choices[0][0].save(lose1_path)  # Image with least num_choices
        image_choices[1][0].save(lose2_path)  # Image with moderate num_choices
        image_choices[2][0].save(lose3_path)  # Image with highest num_choices

            # Append data to CSV
        pd.DataFrame([[full_text, win_path, lose1_path, lose2_path, lose3_path]], columns=columns)\
              .to_csv(output_csv, mode='a', header=False, index=False)
    else:
      print("Problem")


def load_dataset_parallel():
    dataset = load_dataset(
        'poloclub/diffusiondb', 
        'large_random_1k',          # Changed to 'large_random_20k'
        num_proc=os.cpu_count(),  # Use all available CPU cores
        cache_dir="./dataset_cache",  # Cache the dataset locally
        # split='train'
    )
    
    # Convert to pandas more efficiently
    external_df = dataset['train'].to_pandas()
    # external_df.to_csv('external_df.csv',index=False)
    return external_df

######## Also this code you can just replace

if __name__ == '__main__':
    print(time.time())
    # dataset = load_dataset('poloclub/diffusiondb', 'large_random_100k')
    # external_df=dataset['train'].to_pandas()
    external_df=load_dataset_parallel()
    print(time.time())
    output_csv = 'final_dataset.csv'
    columns = ["prompt", "win", "lose1", "lose2", "lose3"]
    pd.DataFrame(columns=columns).to_csv(output_csv, index=False)

    prompt_list=[]
    with open(r'/mnt/home-ldap/bansal_ldap/pulkit/data_100k.txt') as f:
        lines = f.readlines()
        for line in lines:
            prompt_list.append(line.strip())

    data_generation1(type=0,prompt_list=prompt_list,external_df=None,index=0)
    data_generation1(type=1,prompt_list=prompt_list,external_df=external_df,index=num_samples)

    print(time.time())


# prompt_list=[]
# with open('eval_prompt2.txt') as f:
#     lines = f.readlines()
#     for line in lines:
#         prompt_list.append(line.strip())

# data_generation2(type=2,prompt_list=prompt_list,external_df=None,index=2000)
# data_generation2(type=3,prompt_list=prompt_list,external_df=external_df,index=3000)

    final_df = pd.read_csv(output_csv)  # Removed 'index=False' from read_csv
    df_easy = final_df[["prompt", "win", "lose3"]]
    df_medium = final_df[["prompt", "win", "lose2"]]
    df_hard = final_df[["prompt", "win", "lose1"]]

    df_easy.to_csv('df_easy.csv', index=False)
    df_medium.to_csv('df_medium.csv', index=False)
    df_hard.to_csv('df_hard.csv', index=False)
