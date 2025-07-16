import cv2
import numpy as np
import random
import pandas as pd
from PIL import Image, ImageOps
import os
from datasets import load_dataset
import matplotlib.pyplot as plt
import cv2
import textwrap

from skimage import transform, img_as_ubyte
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import numpy as np
import random
import json
from datasets import load_dataset
import ImageReward as RM
from concurrent.futures import ThreadPoolExecutor, as_completed,ProcessPoolExecutor
import cv2
import numpy as np
from skimage import transform
from skimage.util import img_as_ubyte


# --- Distortion Function Definitions (Reduced Intensity) ---

def apply_gaussian_blur(image_np): # Renamed back from 'heavy'
    """Applies Gaussian Blur with a moderate kernel size."""
    # Reduced kernel size range
    ksize = random.randrange(3, 15, 2) # Random odd kernel size between 3 and 13
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), 0)
    return blurred_image


def add_gaussian_noise(image_np): # Renamed back from 'heavy'
    """Adds moderate Gaussian noise."""
    mean = 0
    std_dev = random.uniform(5, 40) # Reduced standard deviation range
    (h, w) = image_np.shape[:2]
    if len(image_np.shape) == 3:
        noise = np.random.normal(mean, std_dev, (h, w, 3))
    else:
        noise = np.random.normal(mean, std_dev, (h, w))
    noisy_image = np.clip(image_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_shear(image_np):
    """Applies shear transformation, replicating borders."""
    (h, w) = image_np.shape[:2]
    shear_factor_x = random.uniform(-0.25, 0.25) # Reduced shear range
    shear_factor_y = random.uniform(-0.25, 0.25)
    M = np.array([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]], dtype=np.float32)
    # Use replicate border instead of black fill
    sheared_image = cv2.warpAffine(image_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return sheared_image

def add_salt_pepper_noise(image_np):
    """Adds salt and pepper noise at lower density."""
    (h, w) = image_np.shape[:2]
    amount = random.uniform(0.002, 0.05) # Reduced amount (0.2% to 3%)
    num_salt = np.ceil(amount * image_np.size * 0.5)
    num_pepper = np.ceil(amount * image_np.size * 0.5)
    noisy_image = image_np.copy()
    # Salt
    coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image_np.shape[:2]]
    # Pepper
    coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_np.shape[:2]]

    if len(image_np.shape) == 3:
        noisy_image[coords_salt[0], coords_salt[1], :] = (255, 255, 255)
        noisy_image[coords_pepper[0], coords_pepper[1], :] = (0, 0, 0)
    else: # Grayscale
        noisy_image[coords_salt[0], coords_salt[1]] = 255
        noisy_image[coords_pepper[0], coords_pepper[1]] = 0

    return noisy_image


def apply_color_jitter(image_np):
    """Applies small random changes to brightness, contrast, and saturation."""
    image = image_np.astype(np.float32)
    alpha = random.uniform(0.8, 1.6)  # contrast
    beta = random.uniform(-20, 20)   # brightness
    image = image * alpha + beta
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def apply_pixelation(image_np):
    """Pixelates one or more random regions of the image."""
    output_image = image_np.copy()
    (h, w) = image_np.shape[:2]
    num_blocks = random.randint(1, 4)

    for _ in range(num_blocks):
        block_w = random.randint(int(w * 0.1), int(w * 0.3))
        block_h = random.randint(int(h * 0.1), int(h * 0.3))
        x = random.randint(0, max(0, w - block_w - 1))
        y = random.randint(0, max(0, h - block_h - 1))
        region = output_image[y:y + block_h, x:x + block_w]

        pixel_size = random.randint(4, 20)
        temp = cv2.resize(region, (block_w // pixel_size, block_h // pixel_size), interpolation=cv2.INTER_NEAREST)
        pixelated = cv2.resize(temp, (block_w, block_h), interpolation=cv2.INTER_NEAREST)
        output_image[y:y + block_h, x:x + block_w] = pixelated

    return output_image

def erase_with_inpainting(image_np):
    """Erases random small regions and fills them using surrounding content."""
    output_image = image_np.copy()
    (h, w) = image_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)  # inpaint mask

    num_occlusions = random.randint(1, 3)

    for _ in range(num_occlusions):
        shape_type = random.choice(['rectangle', 'circle'])

        if shape_type == 'rectangle':
            occ_w = random.randint(int(w * 0.04), int(w * 0.25))
            occ_h = random.randint(int(h * 0.04), int(h * 0.25))
            occ_x = random.randint(0, max(0, w - occ_w - 1))
            occ_y = random.randint(0, max(0, h - occ_h - 1))
            cv2.rectangle(mask, (occ_x, occ_y), (occ_x + occ_w, occ_y + occ_h), 255, -1)
        else:  # circle
            radius = random.randint(int(min(w, h) * 0.02), int(min(w, h) * 0.12))
            center_x = random.randint(radius, max(radius + 1, w - radius - 1))
            center_y = random.randint(radius, max(radius + 1, h - radius - 1))
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Inpaint the masked regions using Telea algorithm (alternative: cv2.INPAINT_NS)
    inpainted = cv2.inpaint(output_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return inpainted

def apply_elastic_transform(image_np): # Renamed back from 'strong'
    """Applies elastic deformation with moderate intensity."""
    alpha = random.uniform(30, 80) # Reduced intensity range
    img_size = max(image_np.shape[0], image_np.shape[1])
    # Sigma adjusted relative to alpha
    sigma = random.uniform(max(3, img_size * 0.01), max(5, img_size * 0.03))

    shape = image_np.shape
    shape_size = shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(*shape_size) * 2 - 1), (0,0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape_size) * 2 - 1), (0,0), sigma) * alpha
    dx = dx.astype(np.float32)
    dy = dy.astype(np.float32)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    map_x = x + dx
    map_y = y + dy
    distorted_image = cv2.remap(image_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return distorted_image



def apply_posterize(image_np):
    """Reduces the number of bits, but less drastically."""
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    bits = random.randint(1, 6) # Reduced to 8-64 colors per channel (more bits = less effect)
    posterized_pil = ImageOps.posterize(image_pil, bits)
    output_image_np = cv2.cvtColor(np.array(posterized_pil), cv2.COLOR_RGB2BGR)
    return output_image_np

def simulate_jpeg_compression(image_np):
    """Simulates saving and reloading as a medium-quality JPEG."""
    quality = random.randint(1, 40) # Higher quality range (less compression artifact)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_img = cv2.imencode('.jpg', image_np, encode_param)

    if not result:
        print("Error: Failed to encode image for JPEG simulation.")
        return image_np
    decoded_img = cv2.imdecode(encoded_img, 1)
    if decoded_img is None:
         print("Error: Failed to decode image after JPEG simulation.")
         return image_np

    
    return decoded_img

def apply_channel_swap(image_np):
    """Randomly swaps channels (less likely to drop)."""
    if len(image_np.shape) < 3:
        # Image is already grayscale or has no channels to swap
        return image_np

    output_image = image_np.copy()
    # Convert the tuple returned by split into a mutable list
    channels = list(cv2.split(output_image))

    # Make swapping more likely than dropping
    action = random.choices(['swap', 'drop', 'swap_bw'], weights=[0.7, 0.15, 0.15], k=1)[0]

    if action == 'swap':
        # Now shuffle works because channels is a list
        random.shuffle(channels)
        output_image = cv2.merge(channels) # cv2.merge can take a list or tuple
    elif action == 'drop':
        channel_to_drop = random.randint(0, 2)
        zero_channel = np.zeros_like(channels[0])
        # Now item assignment works because channels is a list
        channels[channel_to_drop] = zero_channel
        output_image = cv2.merge(channels)
    elif action == 'swap_bw':
         # No modification needed here, just accessing elements
         ch_idx = random.sample(range(3), 3)
         gray_equiv = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
         # Merging selected channels (original or grayscale)
         output_image = cv2.merge((channels[ch_idx[0]], channels[ch_idx[1]], gray_equiv))

    return output_image

# === Helper Function (inside distortions) ===
def _generate_random_mask(img_shape, num_points=10):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    h, w = img_shape[:2]
    points = np.random.randint(0, min(w, h), size=(num_points, 2))
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, points, 255)
    return mask

def _blend_with_mask(original, distorted, mask, blur_size=21):
    # Normalize blurred mask to range [0, 1] for alpha blending
    blurred_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    alpha = blurred_mask.astype(np.float32) / 255.0

    # Ensure 3 channels
    if len(original.shape) == 2 or original.shape[2] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        distorted = cv2.cvtColor(distorted, cv2.COLOR_GRAY2BGR)

    # Expand alpha to 3 channels
    alpha = cv2.merge([alpha] * 3)

    # Blend
    blended = (alpha * distorted + (1 - alpha) * original).astype(np.uint8)
    return blended


# === 1. Swirl Distortion ===
def apply_swirl(img):
    strength = random.uniform(10, 20)  # Reduced strength for less distortion
    radius= random.uniform(100, 300)  # Reduced radius for less distortion
    img_norm = img / 255.0

    # Generate random mask
    mask = _generate_random_mask(img.shape, num_points=np.random.randint(6, 15))

    # Compute center of mass of the mask (as swirl center)
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        swirl_center = (cx, cy)
    else:
        swirl_center = (img.shape[1] // 2, img.shape[0] // 2)  # fallback to center

    # Apply swirl at computed center
    swirled = transform.swirl(img_norm, strength=strength, radius=radius, center=swirl_center)
    swirled = img_as_ubyte(swirled)

    return _blend_with_mask(img, swirled, mask)


# === 2. Sine Wave Distortion ===
def sine_wave_distortion(img, amplitude=20, wavelength=50):
    rows, cols = img.shape[:2]
    y_indices = np.arange(rows)
    shifts = (amplitude * np.sin(2 * np.pi * y_indices / wavelength)).astype(int)
    x_coords = np.arange(cols)
    map_x = np.zeros((rows, cols), dtype=np.float32)
    for y in range(rows):
        map_x[y, :] = (x_coords - shifts[y]) % cols
    map_y = np.repeat(np.arange(rows)[:, np.newaxis], cols, axis=1).astype(np.float32)
    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    mask = _generate_random_mask(img.shape, num_points=np.random.randint(6, 15))
    return _blend_with_mask(img, distorted, mask)

# === 3. Twist (Polar Rotation) Distortion ===
def twist_distortion(img, strength=5.0):
    rows, cols = img.shape[:2]
    center_x, center_y = cols / 2, rows / 2
    step = max(1, min(rows, cols) // 150)
    y_indices = np.arange(0, rows, step)
    x_indices = np.arange(0, cols, step)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    dx = x_grid - center_x
    dy = y_grid - center_y
    radius = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx) + strength * np.exp(-radius / 200)
    new_x = center_x + radius * np.cos(angle)
    new_y = center_y + radius * np.sin(angle)
    map_x = cv2.resize(new_x, (cols, rows)).astype(np.float32)
    map_y = cv2.resize(new_y, (cols, rows)).astype(np.float32)
    map_x = np.clip(map_x, 0, cols - 1)
    map_y = np.clip(map_y, 0, rows - 1)
    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    mask = _generate_random_mask(img.shape, num_points=np.random.randint(6, 15))
    return _blend_with_mask(img, distorted, mask)

# === 4. Radial Zoom Distortion ===
def radial_zoom(img, factor=0.001):
    rows, cols = img.shape[:2]
    cx, cy = cols / 2, rows / 2
    step = max(1, min(rows, cols) // 200)
    y_indices = np.arange(0, rows, step)
    x_indices = np.arange(0, cols, step)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    dx = x_grid - cx
    dy = y_grid - cy
    r = np.sqrt(dx * dx + dy * dy)
    map_x_small = cx + dx * (1 + factor * r)
    map_y_small = cy + dy * (1 + factor * r)
    map_x = cv2.resize(map_x_small, (cols, rows)).astype(np.float32)
    map_y = cv2.resize(map_y_small, (cols, rows)).astype(np.float32)
    map_x = np.clip(map_x, 0, cols - 1)
    map_y = np.clip(map_y, 0, rows - 1)
    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    mask = _generate_random_mask(img.shape, num_points=np.random.randint(6, 15))
    return _blend_with_mask(img, distorted, mask)


def apply_distortion(selected_distortion_functions, distorted_image):
    
    for i, distortion_func in enumerate(selected_distortion_functions):
            
            try:
                # print(type(distorted_image))
                # print(distortion_func)
                temp_distorted_image = distortion_func(distorted_image)
                if temp_distorted_image is not None:
                    distorted_image = temp_distorted_image
                
            except Exception as e:
                print(e)
                # return None
                
    
    return distorted_image

import random

def generate_sequence(min_val, max_val, total_count):
    sequence = []
    remaining = total_count
    
    values = list(range(min_val, max_val + 1))
    random.shuffle(values)
    
    for value in values:
        if remaining <= 0:
            break
        
        if remaining == 1:
            count = 1
        else:
            max_count = min(remaining, remaining // len([v for v in values if v >= value]) + 1)
            count = random.randint(1, max(1, max_count))
        
        sequence.extend([value] * count)
        remaining -= count
    
    while remaining > 0:
        value = random.choice(values)
        sequence.append(value)
        remaining -= 1
    
    return sorted(sequence)



if __name__ == "__main__":
    
    # # Pulkit- Just change the start and end values
    start=0
    end=10
    type_of_process=1 
    min_val=3
    max_val=11
    total_sample=150
    batch_size=16

    output_dir = "dataset"
    final_csv_path=os.path.join(output_dir,f"final_dataset_{start}_{end}")
    final_json_path=os.path.join(output_dir,f"scores_data_{start}_{end}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "win"))
        for n in range(1, batch_size+1):  
            os.mkdir(os.path.join(output_dir, f"lose{n}"))


    # Load the dataset from Hugging Face
    dataset = load_dataset("data-is-better-together/open-image-preferences-v1-binarized")
    df = pd.DataFrame(dataset['train'])

    available_distortions = [
        apply_gaussian_blur,
        # apply_rotation,
        # apply_partial_rotation,
        add_gaussian_noise,
        apply_shear,
        add_salt_pepper_noise,
        apply_pixelation,
        apply_elastic_transform,
        apply_posterize,
        simulate_jpeg_compression,
        apply_channel_swap,
        apply_color_jitter,  
        erase_with_inpainting,
        apply_swirl,
        sine_wave_distortion,
        twist_distortion,
        radial_zoom
    ]

    IM = RM.load("ImageReward-v1.0")
    lose_cols = [f"lose_image{i}" for i in range(1, batch_size+1)]
    final_dataset = pd.DataFrame(columns=["prompt", "win_image"] + lose_cols)

    json_dict_for_scores=[]


    for i in range(start,end):

        temp_dict={}
        if i % 100 == 0:
            print(f"Processing row {i}")
        prompt = df["prompt"][i]

        try:
            nparr = np.frombuffer(df["chosen"][i]['bytes'], np.uint8)
            win_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
            nparr = np.frombuffer(df["rejected"][i]['bytes'], np.uint8)
            lose_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if win_image is None or lose_image is None:
                continue

            distorted_images = []

            # Generate 100 distorted images
            import time
            start=time.time()
            
            if type_of_process==2:
                seq=generate_sequence(min_val,max_val,total_sample)

            def generate_distorted(ind):
                # print(ind)

                base_image = random.choice([win_image, lose_image])
                if type_of_process==2:
                    num_ops = random.randint(seq[ind], len(available_distortions))
                else:
                    num_ops = random.randint(3, len(available_distortions))

                funcs = random.choices(available_distortions, k=num_ops)
                distorted = apply_distortion(funcs, base_image)
                if distorted is None:
                    return None
                if distorted.dtype != np.uint8:
                    distorted = np.clip(distorted, 0, 255).astype(np.uint8)
                return distorted

            with ProcessPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(generate_distorted,ind) for ind in range(total_sample)]

                for future in as_completed(futures):
                    distorted = future.result()
                    if distorted is None:
                        continue
                    time_start = time.time()
                    score = IM.score(prompt,Image.fromarray(distorted))
                    distorted_images.append((distorted, score))
                    # print(score)
            
            if i % 200 == 0:  # Note: Use ==, not just `if i % 200`
                save_folder_path=os.path.join(output_dir, f"ckpt_{i}")
                os.makedirs(save_folder_path, exist_ok=True)
                for idx, (img, _) in enumerate(distorted_images):
                    save_path=os.path.join(save_folder_path,f'{idx}.png')
                    cv2.imwrite(save_path, img)


            print(f"Distortion time: {time.time() - start:.2f} seconds")
            def max_variation_dp(data, k):
                from functools import lru_cache

                data = sorted(data, key=lambda x: x[1])
                values = [val for val in data]
                scores = [val[1] for val in data]
                N = len(data)

                # Use indices instead of actual score values to make caching effective
                @lru_cache(maxsize=None)
                def dp(pos, rem, last_idx):
                    if rem == 0:
                        return 0, []
                    if pos == N:
                        return float("-inf"), []

                    # Option 1: Take current element
                    take_score = abs(scores[pos] - scores[last_idx]) if last_idx != -1 else 0
                    take_sum, take_list = dp(pos + 1, rem - 1, pos)
                    take_sum += take_score

                    # Option 2: Skip current element
                    skip_sum, skip_list = dp(pos + 1, rem, last_idx)

                    if take_sum > skip_sum:
                        return take_sum, [values[pos]] + take_list
                    else:
                        return skip_sum, skip_list

                _, best_subset = dp(0, k, -1)
                return best_subset


            distorted_images = sorted(distorted_images, key=lambda x: x[1])
            distorted_images1= distorted_images[:len(distorted_images)//3]
            distorted_images2= distorted_images[len(distorted_images)//3:len(distorted_images)//2]
            distorted_images3= distorted_images[len(distorted_images)//2:]
            sample_from_each_bucket=batch_size//3
            best_subset = max_variation_dp(distorted_images1, k=sample_from_each_bucket)
            best_subset += max_variation_dp(distorted_images2, k=sample_from_each_bucket)
            best_subset+=max_variation_dp(distorted_images3, k=batch_size-2*sample_from_each_bucket)

            best_subset=sorted(best_subset, key=lambda x: x[1])
            if len(best_subset) < batch_size:
                continue

            # Save win image
            win_path = os.path.join(output_dir, "win", f"{i}.png")
            if win_image.dtype != np.uint8:
                win_image = np.clip(win_image, 0, 255).astype(np.uint8)
            cv2.imwrite(win_path, win_image)

            import matplotlib.pyplot as plt

            lose_paths = []
            for j, (img, score_val) in enumerate(best_subset):
                path = os.path.join(output_dir, f"lose{j+1}", f"{i}.png")
                cv2.imwrite(path, img)
                lose_paths.append(path)
                temp_dict[j+1]=score_val

            win_image_score= IM.score(prompt, Image.fromarray(win_image))
            temp_dict['win_image_score']=win_image_score

            #Pulkit- Comment this line to avoid visualization
            # visualize_generated_dataset(best_subset, win_image, prompt,win_image_score)

            # Append to final dataset
            data_row = {"prompt": prompt, "win_image": win_path}
            for k in range(batch_size):
                data_row[f"lose_image{k+1}"] = lose_paths[k]

            final_dataset = pd.concat([final_dataset, pd.DataFrame([data_row])], ignore_index=True)

        except Exception as e:
            print(f"Error on row {i}: {e}")
            continue

        json_dict_for_scores.append({i: temp_dict})
        # if i%100==0:
        with open(final_json_path,'w') as f:
            json.dump(json_dict_for_scores, f, indent=4)
    # Save full dataset
    final_dataset.to_csv(final_csv_path, index=False)

   
        


        
