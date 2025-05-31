import cv2
import numpy as np
import random
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import math
import io
import shutil
import requests
from datasets import load_dataset,concatenate_datasets
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import multiprocessing as mp

# --- Distortion Function Definitions (Reduced Intensity) ---

def apply_gaussian_blur(image_np): # Renamed back from 'heavy'
    """Applies Gaussian Blur with a moderate kernel size."""
    # Reduced kernel size range
    ksize = random.randrange(3, 15, 2) # Random odd kernel size between 3 and 13
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), 0)
    return blurred_image

def apply_rotation(image_np):
    """Rotates the entire image by a random angle, replicating borders."""
    angle = random.uniform(-30, 30) # Reduced angle range
    (h, w) = image_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Changed borderMode back to replicate to avoid black corners
    rotated_image = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def apply_partial_rotation(image_np):
    """Rotates a random rectangular portion, using constant fill for contrast."""
    (h, w) = image_np.shape[:2]
    min_size_ratio = 0.15 # Slightly smaller min
    max_size_ratio = 0.50 # Reduced max size
    roi_w = random.randint(int(w * min_size_ratio), int(w * max_size_ratio))
    roi_h = random.randint(int(h * min_size_ratio), int(h * max_size_ratio))
    roi_x = random.randint(0, max(0, w - roi_w - 1))
    roi_y = random.randint(0, max(0, h - roi_h - 1))
    roi = image_np[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    angle = random.uniform(-90, 90) # Reduced angle range for partial
    (roi_h_actual, roi_w_actual) = roi.shape[:2]
    if roi_h_actual == 0 or roi_w_actual == 0:
        return image_np
    roi_center = (roi_w_actual // 2, roi_h_actual // 2)
    M = cv2.getRotationMatrix2D(roi_center, angle, 1.0)
    # Keep constant border for this one to make the rotated patch distinct, maybe gray?
    fill_color_val = random.randint(50, 200)
    fill_color = (fill_color_val, fill_color_val, fill_color_val)
    if len(image_np.shape) < 3: fill_color = fill_color_val
    rotated_roi = cv2.warpAffine(roi, M, (roi_w_actual, roi_h_actual), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_color)
    output_image = image_np.copy()
    output_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = rotated_roi
    return output_image


def add_gaussian_noise(image_np): # Renamed back from 'heavy'
    """Adds moderate Gaussian noise."""
    mean = 0
    std_dev = random.uniform(5, 25) # Reduced standard deviation range
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
    amount = random.uniform(0.002, 0.03) # Reduced amount (0.2% to 3%)
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

def add_random_occlusion(image_np):
    """Adds fewer/smaller random filled shapes."""
    output_image = image_np.copy()
    (h, w) = image_np.shape[:2]
    num_occlusions = random.randint(1, 3) # Reduced number of occlusions

    for _ in range(num_occlusions):
        color_val = random.randint(0, 255)
        color = (color_val, color_val, color_val)
        if len(image_np.shape) == 3 and random.random() > 0.3:
             color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif len(image_np.shape) < 3:
             color = color_val

        shape_type = random.choice(['rectangle', 'circle'])

        if shape_type == 'rectangle':
            # Reduced max size
            occ_w = random.randint(int(w * 0.04), int(w * 0.25))
            occ_h = random.randint(int(h * 0.04), int(h * 0.25))
            occ_x = random.randint(0, max(0, w - occ_w - 1))
            occ_y = random.randint(0, max(0, h - occ_h - 1))
            cv2.rectangle(output_image, (occ_x, occ_y), (occ_x + occ_w, occ_y + occ_h), color, -1)
        else: # Circle
            # Reduced max radius
            radius = random.randint(int(min(w, h) * 0.02), int(min(w, h) * 0.12))
            center_x = random.randint(radius, max(radius + 1, w - radius - 1))
            center_y = random.randint(radius, max(radius + 1, h - radius - 1))
            cv2.circle(output_image, (center_x, center_y), radius, color, -1)

    return output_image

def apply_pixelation(image_np):
    """Applies light pixelation effect with reduced intensity."""
    (h, w) = image_np.shape[:2]
    
    # Much smaller pixel blocks for subtle effect
    min_pix_size = max(2, min(h, w) // 150)  # Increased divisor from 80 to 150
    max_pix_size = max(min_pix_size + 1, min(h, w) // 50)  # Increased divisor from 15 to 50
    
    if min_pix_size >= max_pix_size: 
        max_pix_size = min_pix_size + 1
    
    pixel_size = random.randint(min_pix_size, max_pix_size)
    if pixel_size <= 0: 
        pixel_size = 2  # Minimum safe value
    
    # Apply pixelation
    temp_img = cv2.resize(image_np, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_NEAREST)
    pixelated_image = cv2.resize(temp_img, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Optional: Blend with original for even more subtle effect
    # Uncomment the next two lines if you want to blend with original image
    # blend_factor = random.uniform(0.3, 0.7)  # 30-70% pixelation
    # pixelated_image = cv2.addWeighted(image_np, 1-blend_factor, pixelated_image, blend_factor, 0)
    
    return pixelated_image

def apply_elastic_transform(image_np): # Renamed back from 'strong'
    """Applies elastic deformation with moderate intensity."""
    alpha = random.uniform(15, 40) # Reduced intensity range
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

def apply_cutout(image_np):
    """Blacks out a smaller random rectangular portion."""
    output_image = image_np.copy()
    (h, w) = image_np.shape[:2]
    # Reduced max cutout size
    min_dim = min(h, w)
    cutout_h = random.randint(int(min_dim * 0.05), int(min_dim * 0.30)) # 5% to 30%
    cutout_w = random.randint(int(min_dim * 0.05), int(min_dim * 0.30))

    cutout_h = min(cutout_h, h - 1)
    cutout_w = min(cutout_w, w - 1)
    if cutout_h <=0 or cutout_w <=0:
        return image_np

    cutout_y = random.randint(0, max(0, h - cutout_h - 1))
    cutout_x = random.randint(0, max(0, w - cutout_w - 1))

    color = (0, 0, 0)
    if len(image_np.shape) < 3: color = 0

    cv2.rectangle(output_image, (cutout_x, cutout_y), (cutout_x + cutout_w, cutout_y + cutout_h), color, -1)
    return output_image

def apply_posterize(image_np):
    """Reduces the number of bits, but less drastically."""
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    bits = random.randint(3, 6) # Reduced to 8-64 colors per channel (more bits = less effect)
    posterized_pil = ImageOps.posterize(image_pil, bits)
    output_image_np = cv2.cvtColor(np.array(posterized_pil), cv2.COLOR_RGB2BGR)
    return output_image_np

def simulate_jpeg_compression(image_np):
    """Simulates saving and reloading as a medium-quality JPEG."""
    quality = random.randint(30, 75) # Higher quality range (less compression artifact)
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

def apply_distortion(selected_distortion_functions, distorted_image):
    
    for i, distortion_func in enumerate(selected_distortion_functions):
            
            try:
                # print(type(distorted_image))
                print(distortion_func)
                temp_distorted_image = distortion_func(distorted_image)
                if temp_distorted_image is not None:
                    distorted_image = temp_distorted_image
                
            except Exception as e:
                print(e)
                # return None
                
    
    return distorted_image

# --- Main Execution Logic ---
def tensor_to_numpy(tensor_img):
    """Convert tensor image back to numpy array in OpenCV format"""
    # tensor_img is expected to be [1, C, H, W] in range [0, 1]
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.squeeze(0)  # Remove batch dimension: [C, H, W]
    
    # Convert from [C, H, W] to [H, W, C]
    numpy_img = tensor_img.permute(1, 2, 0).numpy()
    
    # Convert from [0, 1] to [0, 255] and to uint8
    numpy_img = (numpy_img * 255).astype(np.uint8)
    
    # Convert from RGB to BGR for OpenCV
    if numpy_img.shape[2] == 3:
        numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
    
    return numpy_img

def max_variation_dp(data, k=16):
    """Dynamic programming approach to select k most varied items."""
    from functools import lru_cache
    data = sorted(data, key=lambda x: x[1])
    N = len(data)

    @lru_cache(maxsize=None)
    def dp(pos, rem, last_score):
        if rem == 0:
            return 0, []
        if pos == N:
            return float("-inf"), []

        take_score = abs(data[pos][1] - last_score) if last_score is not None else 0
        take_sum, take_list = dp(pos + 1, rem - 1, data[pos][1])
        take_sum += take_score

        skip_sum, skip_list = dp(pos + 1, rem, last_score)

        if take_sum > skip_sum:
            return take_sum, [data[pos]] + take_list
        else:
            return skip_sum, skip_list

    _, best_subset = dp(0, k, None)
    return best_subset

def process_single_row(args):
    """Process a single row of data - this function will be parallelized."""
    i, row_data, output_dir, available_distortions = args
    
    try:
        # Import required modules inside the function for multiprocessing
        import torch
        import torchvision.transforms as T
        from torchmetrics.multimodal.clip_score import CLIPScore
        
        # Initialize CLIP metric for this process
        clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        
        prompt = row_data["prompt"]
        
        # Decode images
        nparr = np.frombuffer(row_data["chosen"]['bytes'], np.uint8)
        win_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
        nparr = np.frombuffer(row_data["rejected"]['bytes'], np.uint8)
        lose_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if win_image is None or lose_image is None:
            return None

        distorted_images = []

        # Generate 32 distorted images
        for _ in range(32):
            base_image = random.choice([win_image, lose_image])
            num_ops = random.randint(2, len(available_distortions))
            funcs = random.choices(available_distortions, k=num_ops)
            distorted = apply_distortion(funcs, base_image)
            if distorted is None:
                continue
            if distorted.dtype != np.uint8:
                distorted = np.clip(distorted, 0, 255).astype(np.uint8)

            transform = T.Compose([
                T.ToTensor()  # Converts HxWxC in [0, 255] to CxHxW in [0.0, 1.0]
            ])
            distorted_tensor = transform(distorted).unsqueeze(0)
            score = clip_metric(distorted_tensor, prompt)
            distorted_numpy = tensor_to_numpy(distorted_tensor)  # Convert back to numpy for saving
            distorted_images.append((distorted_numpy, score))

        # Select 16 best varied distorted images
        best_subset = max_variation_dp(distorted_images, k=16)
        if len(best_subset) < 16:
            return None

        # Save win image
        win_path = os.path.join(output_dir, "win", f"{i}.png")
        if win_image.dtype != np.uint8:
            win_image = np.clip(win_image, 0, 255).astype(np.uint8)
        cv2.imwrite(win_path, win_image)

        # Save lose images
        lose_paths = []
        for j, (img, _) in enumerate(best_subset):
            path = os.path.join(output_dir, f"lose{j+1}", f"{i}.png")
            cv2.imwrite(path, img)
            lose_paths.append(path)

        # Create data row
        data_row = {"prompt": prompt, "win_image": win_path}
        for k in range(16):
            data_row[f"lose_image{k+1}"] = lose_paths[k]

        return data_row

    except Exception as e:
        print(f"Error processing row {i}: {e}")
        return None

def process_chunk(chunk_args):
    """Process a chunk of rows sequentially within a single process."""
    chunk_data, output_dir, available_distortions = chunk_args
    results = []
    
    for args in chunk_data:
        result = process_single_row((*args, output_dir, available_distortions))
        if result is not None:
            results.append(result)
    
    return results

if __name__ == "__main__":
    # --- Configuration ---
    output_dir = "dataset"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "win"))
        for n in range(1, 17):  # lose1 to lose16
            os.mkdir(os.path.join(output_dir, f"lose{n}"))

    # Load the dataset from Hugging Face
    dataset = load_dataset("data-is-better-together/open-image-preferences-v1-binarized")
    df = pd.DataFrame(dataset['train'])

    available_distortions = [
        apply_gaussian_blur,
        apply_partial_rotation,
        add_gaussian_noise,
        apply_shear,
        add_salt_pepper_noise,
        add_random_occlusion,
        apply_pixelation,
        apply_elastic_transform,
        apply_cutout,
        apply_posterize,
        simulate_jpeg_compression,
        apply_channel_swap,
    ]

    # Prepare data for parallel processing
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU core free
    print(f"Using {num_processes} processes for parallel processing")
    
    # Create chunks of data to reduce memory overhead
    chunk_size = max(1, len(df) // (num_processes * 4))  # Create more chunks than processes
    chunks = []
    
    for start_idx in range(0, len(df), chunk_size):
        end_idx = min(start_idx + chunk_size, len(df))
        chunk_data = []
        for i in range(start_idx, end_idx):
            chunk_data.append((i, df.iloc[i]))
        chunks.append((chunk_data, output_dir, available_distortions))
    
    print(f"Processing {len(df)} rows in {len(chunks)} chunks with {num_processes} processes")

    # Process chunks in parallel
    all_results = []
    with Pool(processes=num_processes) as pool:
        # Use tqdm for progress tracking
        chunk_results = list(tqdm(
            pool.imap(process_chunk, chunks), 
            total=len(chunks), 
            desc="Processing chunks"
        ))
        
        # Flatten results
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)

    # Create final dataset
    final_dataset = pd.DataFrame(all_results)
    
    # Save full dataset
    final_dataset.to_csv("final_dataset.csv", index=False)
    print(f"Saved {len(final_dataset)} processed rows to final_dataset.csv")

    # Save each difficulty level (16 files)
    for k in range(1, 17):
        df_k = final_dataset[["prompt", "win_image", f"lose_image{k}"]]
        df_k.to_csv(f"df_level{k}.csv", index=False)

    print("Processing completed!")
