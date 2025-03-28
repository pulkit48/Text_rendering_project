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

# --- Distortion Function Definitions (Reduced Intensity) ---

def apply_gaussian_blur(image_np): # Renamed back from 'heavy'
    """Applies Gaussian Blur with a moderate kernel size."""
    # Reduced kernel size range
    ksize = random.randrange(3, 15, 2) # Random odd kernel size between 3 and 13
    print(f"Applying Gaussian Blur with kernel size: {ksize}")
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
    print(f"Applying full rotation with angle: {angle:.2f} degrees (replicate borders)")
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
        print("Skipping partial rotation due to zero-sized ROI")
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
    print(f"Applying partial rotation: Angle={angle:.2f}, ROI=({roi_x},{roi_y}, {roi_w}, {roi_h})")
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
    print(f"Adding Gaussian Noise with std dev: {std_dev:.2f}")
    return noisy_image

def apply_shear(image_np):
    """Applies shear transformation, replicating borders."""
    (h, w) = image_np.shape[:2]
    shear_factor_x = random.uniform(-0.25, 0.25) # Reduced shear range
    shear_factor_y = random.uniform(-0.25, 0.25)
    M = np.array([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]], dtype=np.float32)
    # Use replicate border instead of black fill
    sheared_image = cv2.warpAffine(image_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    print(f"Applying Shear with factors: x={shear_factor_x:.2f}, y={shear_factor_y:.2f}")
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

    print(f"Adding Salt & Pepper noise with amount: {amount*100:.2f}%")
    return noisy_image

def add_random_occlusion(image_np):
    """Adds fewer/smaller random filled shapes."""
    output_image = image_np.copy()
    (h, w) = image_np.shape[:2]
    num_occlusions = random.randint(1, 3) # Reduced number of occlusions
    print(f"Adding {num_occlusions} random occlusions.")

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

def apply_pixelation(image_np): # Renamed back from 'heavy'
    """Applies moderate pixelation effect."""
    (h, w) = image_np.shape[:2]
    # Reduced max pixel size
    min_pix_size = max(2, min(h, w) // 80) # Small min block size
    max_pix_size = max(min_pix_size + 3, min(h, w) // 15) # Moderate max block size
    if min_pix_size >= max_pix_size: max_pix_size = min_pix_size + 2

    pixel_size = random.randint(min_pix_size, max_pix_size)
    if pixel_size <= 0: pixel_size = min_pix_size

    temp_img = cv2.resize(image_np, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_NEAREST)
    pixelated_image = cv2.resize(temp_img, (w, h), interpolation=cv2.INTER_NEAREST)
    print(f"Applying Pixelation with block size approx {pixel_size}x{pixel_size}")
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
    print(f"Applying Elastic Transform with alpha={alpha:.1f}, sigma={sigma:.1f}")
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
        print("Skipping cutout due to small image size.")
        return image_np

    cutout_y = random.randint(0, max(0, h - cutout_h - 1))
    cutout_x = random.randint(0, max(0, w - cutout_w - 1))

    color = (0, 0, 0)
    if len(image_np.shape) < 3: color = 0

    cv2.rectangle(output_image, (cutout_x, cutout_y), (cutout_x + cutout_w, cutout_y + cutout_h), color, -1)
    print(f"Applying Cutout at ({cutout_x},{cutout_y}) size {cutout_w}x{cutout_h}")
    return output_image

def apply_posterize(image_np):
    """Reduces the number of bits, but less drastically."""
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    bits = random.randint(3, 6) # Reduced to 8-64 colors per channel (more bits = less effect)
    posterized_pil = ImageOps.posterize(image_pil, bits)
    print(f"Applying Posterization with {bits} bits per channel")
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

    print(f"Simulating JPEG Compression with quality={quality}")
    return decoded_img

def apply_channel_swap(image_np):
    """Randomly swaps channels (less likely to drop)."""
    if len(image_np.shape) < 3:
        print("Skipping channel swap for grayscale image.")
        return image_np

    output_image = image_np.copy()
    channels = cv2.split(output_image)
    # Make swapping more likely than dropping
    action = random.choices(['swap', 'drop', 'swap_bw'], weights=[0.7, 0.15, 0.15], k=1)[0]

    if action == 'swap':
        random.shuffle(channels)
        output_image = cv2.merge(channels)
        print("Applying Channel Swap (Random Order)")
    elif action == 'drop':
        channel_to_drop = random.randint(0, 2)
        zero_channel = np.zeros_like(channels[0])
        channels[channel_to_drop] = zero_channel
        # Less likely to drop a second channel
        # if random.random() < 0.1:
        #      remaining_channels = [i for i in range(3) if i != channel_to_drop]
        #      if remaining_channels: channels[random.choice(remaining_channels)] = zero_channel
        output_image = cv2.merge(channels)
        print(f"Applying Channel Drop (Dropped channel {channel_to_drop})")
    elif action == 'swap_bw':
         ch_idx = random.sample(range(3), 3)
         gray_equiv = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
         output_image = cv2.merge((channels[ch_idx[0]], channels[ch_idx[1]], gray_equiv))
         print("Applying Channel Swap (mixed with grayscale)")

    return output_image

def apply_distortion(selected_distortion_functions, distorted_image):
    
    for i, distortion_func in enumerate(selected_distortion_functions):
            
            try:
                distorted_image = distortion_func(distorted_image)
                
            except Exception as e:
                return None
                
    
    return distorted_image
    
def url_to_cv2_image(url):
    """Fetch an image from a URL and convert it to an OpenCV image."""
    print(url)
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we got a valid response
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)    
                
# --- Main Execution Logic ---

if __name__ == "__main__":
    # --- Configuration ---
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")
        os.mkdir("dataset")
        os.mkdir("dataset/win")
        os.mkdir("dataset/lose1")
        os.mkdir("dataset/lose2")
        os.mkdir("dataset/lose3")  
    
        
    outut_dir="dataset"
    # df= pd.read_csv(r"C:\Users\F-PulkitB\Downloads\Text_Rendering\pickapic_v2_no_images.csv")
    df=load_dataset("yuvalkirstain/pickapic_v2",num_proc=4)
    merged_dataset = concatenate_datasets([
    df["train"],
    df["validation"],
    df["test"]
    ])
    df=merged_dataset.to_pandas()[['caption','jpg_0','jpg_1',"label_0","label_1"]]
    df=df[df['label_0']!=df['label_1']]
    print(len(df))
    swap_mask = (df['label_0'] == 1) & (df['label_1'] == 0)

    df.loc[swap_mask, ['jpg_0', 'jpg_1']] = df.loc[swap_mask, ['jpg_1', 'jpg_0']].values
    df.to_csv("pickapic_v2_no_images.csv",index=False)
    
    # List of available distortion functions (with reduced intensity parameters)
    available_distortions = [
        apply_gaussian_blur,
        apply_rotation,
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

    final_dataset=pd.DataFrame(columns=['prompt','win_image','lose_image1','lose_image2','lose_image3'])

    for i in range(len(df)):
        print(i)
        prompt = df['caption'][i]
        try:
            # win_image = url_to_cv2_image(df['image_0_url'][i])
            # lose_image = url_to_cv2_image(df['image_1_url'][i])

            # # Convert from BGR (OpenCV default) to RGB if needed
            # win_image = cv2.cvtColor(win_image, cv2.COLOR_BGR2RGB)
            # lose_image = cv2.cvtColor(lose_image, cv2.COLOR_BGR2RGB)

            win_image=cv2.imread(df['jpg_0'][i])
            lose_image1=cv2.imread(df['jpg_1'][i])

            working_image=lose_image1

            # --- Apply Random Distortions (Reduced Number) ---
            max_distortions_available = len(available_distortions)
            # Apply fewer operations overall
            min_ops = 3 # Apply at least 1 operation
            max_ops = 6 # Apply at most 4 operations (adjust as needed)


            num_distortions_to_apply = random.randint(min_ops, max_ops)
            # selected_distortion_functions = random.choices(available_distortions, k=num_distortions_to_apply)
            # lose_image1=apply_distortion(selected_distortion_functions, working_image)

            selected_distortion_functions1 = random.choices(available_distortions, k=num_distortions_to_apply)
            lose_image1=apply_distortion(selected_distortion_functions1, working_image)

            selected_distortion_functions2 = random.choices(available_distortions, k=num_distortions_to_apply)
            lose_image2=apply_distortion(selected_distortion_functions2, working_image)
            
            win_image_path=os.path.join(outut_dir,"win_"+str(i)+".png")
            lose_image1_path=os.path.join(outut_dir,"lose1_"+str(i)+".png")

            
            lose_image2_path=os.path.join(outut_dir,"lose2_"+str(i)+".png")
            lose_image3_path=os.path.join(outut_dir,"lose3_"+str(i)+".png")
            

            if win_image is None or lose_image1 is None or lose_image2 is None or lose_image3 is None:
                continue

            if win_image.dtype != np.uint8: 
                win_image = np.clip(win_image, 0, 255).astype(np.uint8)
            
            if lose_image1.dtype != np.uint8:
                lose_image1 = np.clip(lose_image1, 0, 255).astype(np.uint8)
            
            if lose_image2.dtype != np.uint8:
                lose_image2 = np.clip(lose_image2, 0, 255).astype(np.uint8)
            
            if lose_image3.dtype != np.uint8:
                lose_image3 = np.clip(lose_image3, 0, 255).astype(np.uint8)

            cv2.imwrite(win_image_path, win_image)
            cv2.imwrite(lose_image1_path, lose_image1)

            if selected_distortion_functions1<selected_distortion_functions2:
                cv2.imwrite(lose_image1_path, lose_image1)
                cv2.imwrite(lose_image2_path, lose_image2)
            else:
                cv2.imwrite(lose_image2_path, lose_image2)
                cv2.imwrite(lose_image1_path, lose_image1)

            final_dataset=final_dataset.append({'prompt':prompt,'win_image':win_image_path,'lose_image1':lose_image1_path,'lose_image2':lose_image2_path,'lose_image3':lose_image3_path},ignore_index=True)
        except Exception as e:
            print(e)
            print("Error-------")
            continue

    
    final_dataset.to_csv("final_dataset.csv",index=False)
    df_easy=final_dataset[final_dataset['prompt','win_image','lose_image1']]
    df_medium=final_dataset[final_dataset['prompt','win_image','lose_image2']]
    df_hard=final_dataset[final_dataset['prompt','win_image','lose_image3']]

    df_easy.to_csv("df_easy.csv",index=False)
    df_medium.to_csv("df_medium.csv",index=False)
    df_hard.to_csv("df_hard.csv",index=False)
        


        
