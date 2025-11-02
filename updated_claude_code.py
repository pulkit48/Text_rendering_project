import os
import random
import json
import pandas as pd
from io import BytesIO
from PIL import Image
from functools import lru_cache

# ===========================
# DISTORTION FUNCTIONS
# ===========================

def char_level_repetition_distortion(text: str, max_repeats: int = 2):
    """Randomly repeats characters at random positions."""
    num_positions = random.randint(1, min(4, len(text)))
    random_indices = random.sample(range(len(text)), k=num_positions)

    distorted = ""
    for i, ch in enumerate(text):
        distorted += ch
        if i in random_indices:
            repeat_count = random.randint(1, max_repeats)
            distorted += ch * repeat_count
    return distorted


def char_level_drop_distortion(text: str, max_drops: int = 3):
    """Drops random characters from the text."""
    num_drops = random.randint(1, min(max_drops, len(text) // 2))
    drop_indices = set(random.sample(range(len(text)), k=num_drops))
    distorted = "".join(ch for i, ch in enumerate(text) if i not in drop_indices)
    return distorted


def adjacent_char_swap_distortion(text: str, max_swaps: int = 2):
    """Swaps adjacent alphanumeric characters at random positions."""
    text_list = list(text)
    if len(text_list) < 2:
        return text

    valid_indices = [
        i for i in range(len(text_list) - 1)
        if text_list[i].isalnum() and text_list[i+1].isalnum()
    ]
    
    if not valid_indices:
        return text

    num_swaps = random.randint(1, min(max_swaps, len(valid_indices)))
    swap_indices = random.sample(valid_indices, k=num_swaps)
    
    for idx in swap_indices:
        if text_list[idx].isalnum() and text_list[idx+1].isalnum():
            text_list[idx], text_list[idx+1] = text_list[idx+1], text_list[idx]
            
    return "".join(text_list)


def mirror_distortion(text):
    """Applies mirror distortion to random characters."""
    with open("mirror_distortion.json", "r", encoding="utf-8") as f:
        mirror_data = json.load(f)
        
    number = random.randint(0, len(text) // 2)
    random_index = random.sample(range(len(text)), k=number)
    text_copy = list(text)
    
    mirror_types = ['HORIZONTAL_MIRROR_MULTI', 'VERTICAL_MIRROR_MULTI', 'ROTATION_180_MULTI']
    
    for idx in random_index:
        working_char = text[idx]
        attempt = 0
        
        while attempt < 10:
            selected_type = random.choice(mirror_types)
            if working_char not in mirror_data[selected_type]:
                attempt += 1
                continue

            replacement = random.choice(mirror_data[selected_type][working_char])
            text_copy[idx] = replacement
            break
            
    return ''.join(text_copy)


def same_char_distortion(text):
    """Replaces characters with visually similar ones."""
    with open("same_char.json", "r", encoding="utf-8") as f:
        same_char_data = json.load(f)
    
    number = random.randint(0, len(text) // 2)
    random_index = random.sample(range(len(text)), k=number)
    text_copy = list(text)
    
    for idx in random_index:
        working_char = text[idx]
        if working_char in same_char_data:
            replacement = random.choice(same_char_data[working_char])
            text_copy[idx] = replacement
            break
    
    return ''.join(text_copy)


def case_shuffle_distortion(text):
    """Randomly shuffles case of all characters."""
    distorted = ""
    for ch in text:
        if ch.isalpha():
            distorted += ch.upper() if random.random() < 0.5 else ch.lower()
        else:
            distorted += ch
    return distorted


def noise_injection_distortion(text, max_noise: int = 5):
    """Injects random noise characters at random positions."""
    noise_chars = ['·', '˙', '`', '´', '¨', '˚', '°']
    text_list = list(text)
    num_noise = random.randint(1, min(max_noise, len(text)))
    
    for _ in range(num_noise):
        idx = random.randint(0, len(text_list))
        text_list.insert(idx, random.choice(noise_chars))
    
    return ''.join(text_list)


def ocr_confusion_distortion(text: str, max_confusions: int = 2):
    """Applies OCR-like character confusions."""
    ocr_pairs = {
        'rn': 'm', 'nn': 'u', 'vv': 'w', 'uu': 'w', 'ii': 'u',
        'cl': 'd', 'li': 'h', 'Il': 'H', 'ln': 'h', 'rr': 'n',
        'm': 'rn', 'w': 'vv', 'u': 'ii', 'n': 'ri', 'h': 'li',
        'M': 'RN', 'W': 'VV', 'H': 'tt', 'N': 'AI',
    }
    
    for _ in range(max_confusions):
        for pattern, replacement in ocr_pairs.items():
            if pattern in text and random.random() < 0.3:
                positions = [i for i in range(len(text) - len(pattern) + 1) 
                           if text[i:i+len(pattern)] == pattern]
                if positions:
                    idx = random.choice(positions)
                    text = text[:idx] + replacement + text[idx+len(pattern):]
                    break
    return text


def subscript_superscript_distortion(text: str, max_conversions: int = 2):
    """Converts characters to subscript/superscript."""
    superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                   '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                   'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ'}
    
    subscripts = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                 '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
    
    conversion_map = {**superscripts, **subscripts}
    text_list = list(text)
    valid_indices = [i for i, ch in enumerate(text) if ch in conversion_map]
    
    if not valid_indices:
        return text
    
    num_conversions = random.randint(1, min(max_conversions, len(valid_indices)))
    conversion_indices = random.sample(valid_indices, k=num_conversions)
    
    for idx in conversion_indices:
        text_list[idx] = conversion_map[text[idx]]
    
    return ''.join(text_list)


def zalgo_distortion(text: str, max_intensity: int = 3, max_chars: int = 5):
    """Adds combining diacritic marks to random characters."""
    if len(text) <= 8:
        return text
    
    DIACRITICS = [
        '\u0300', '\u0301', '\u0302', '\u0303', '\u0304', '\u0305', '\u0306', '\u0307', 
        '\u0308', '\u030A', '\u030B', '\u030C', '\u030D', '\u030E', '\u030F', '\u0310', 
        '\u0311', '\u0334', '\u0335', '\u0336', '\u0337', '\u0338',
        '\u0316', '\u0317', '\u0318', '\u0319', '\u031A', '\u031B', '\u031C', '\u031D',
        '\u031E', '\u031F', '\u0320', '\u0321', '\u0322', '\u0323', '\u0324', '\u0325',
        '\u0326', '\u0327', '\u0328', '\u0329', '\u032A'
    ]
    
    text_list = list(text)
    valid_indices = [i for i, char in enumerate(text) if not char.isspace()]
    
    if not valid_indices:
        return text

    num_chars_to_distort = random.randint(1, min(max_chars, len(valid_indices)))
    distort_indices = random.sample(valid_indices, k=num_chars_to_distort)
    
    for idx in sorted(distort_indices, reverse=True):
        num_diacritics = random.randint(1, max_intensity)
        for _ in range(num_diacritics):
            text_list.insert(idx + 1, random.choice(DIACRITICS))
    
    return "".join(text_list)


# ===========================
# TEXT DISTORTION HANDLER
# ===========================

def apply_single_distortion(original_text: str, distortion_func) -> str:
    """
    Applies a SINGLE distortion to the original text.
    This ensures each distortion starts from the clean original text.
    """
    if not original_text or original_text == '':
        return original_text
    
    return distortion_func(original_text)


def distort_text_list(text_list: list, distortion_func) -> list:
    """
    Applies a distortion function to each text in the list.
    Each text is distorted independently from its ORIGINAL state.
    """
    return [apply_single_distortion(text, distortion_func) for text in text_list]


def distort_example(example: dict, distortion_func) -> dict:
    """
    Creates a distorted copy of the example with ONE distortion applied.
    The original example is NOT modified.
    """
    distorted_example = example.copy()
    original_texts = example['text'].copy() if isinstance(example['text'], list) else [example['text']]
    
    # Apply distortion to each text independently
    distorted_texts = distort_text_list(original_texts, distortion_func)
    distorted_example['text'] = distorted_texts
    
    return distorted_example


# ===========================
# SCORING FUNCTION
# ===========================

def score(image, input_text, max_new_tokens=100):
    """
    Placeholder scoring function.
    Replace with actual OCR + Levenshtein distance implementation.
    """
    return random.randint(0, 100)


# ===========================
# DP OPTIMIZATION
# ===========================

def max_variation_dp(data, k):
    """
    Selects k elements with maximum score variation using dynamic programming.
    """
    data = sorted(data, key=lambda x: x[1])
    values = [val for val in data]
    scores = [val[1] for val in data]
    N = len(data)

    @lru_cache(maxsize=None)
    def dp(pos, rem, last_idx):
        if rem == 0:
            return 0, []
        if pos == N:
            return float("-inf"), []

        # Take current element
        take_score = abs(scores[pos] - scores[last_idx]) if last_idx != -1 else 0
        take_sum, take_list = dp(pos + 1, rem - 1, pos)
        take_sum += take_score

        # Skip current element
        skip_sum, skip_list = dp(pos + 1, rem, last_idx)

        if take_sum > skip_sum:
            return take_sum, [values[pos]] + take_list
        else:
            return skip_sum, skip_list

    _, best_subset = dp(0, k, -1)
    return best_subset


# ===========================
# MAIN GENERATION LOGIC
# ===========================

def generate_single_distorted_image(original_example: dict, distortion_func, renderer, ind: int):
    """
    Generates a single distorted image from the original example.
    Each call starts fresh from the original example.
    """
    # Create distorted version (does NOT modify original)
    distorted_example = distort_example(original_example, distortion_func)
    
    print(f"Sample {ind} - Distorted text: {distorted_example['text']}")
    
    # Render the distorted example
    distorted_bytes = renderer.render(distorted_example)
    distorted_image = Image.open(BytesIO(distorted_bytes))
    
    return distorted_image


def generate_dataset(start: int, end: int, dataset, renderer, 
                     batch_size: int = 16, total_samples: int = 150,
                     output_dir: str = "dataset"):
    """
    Main function to generate the dataset with distorted images.
    """
    # Available distortions
    available_distortions = [
        char_level_drop_distortion,
        char_level_repetition_distortion,
        adjacent_char_swap_distortion,
        case_shuffle_distortion,
        noise_injection_distortion,
        ocr_confusion_distortion,
        subscript_superscript_distortion,
        zalgo_distortion,
        mirror_distortion,
        same_char_distortion,
    ]
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "win"), exist_ok=True)
    for n in range(1, batch_size + 1):
        os.makedirs(os.path.join(output_dir, f"lose{n}"), exist_ok=True)
    
    # Initialize dataframe
    lose_cols = [f"lose_image{i}" for i in range(1, batch_size + 1)]
    final_dataset = pd.DataFrame(columns=["prompt", "win_image"] + lose_cols)
    json_dict_for_scores = []
    
    # Process each example
    for i in range(start, end):
        print(f"\n{'='*50}")
        print(f"Processing example {i}/{end}")
        print(f"{'='*50}")
        
        # Get original example
        original_example = dataset[i]
        
        # Render original (win) image
        win_image_bytes = renderer.render(original_example)
        win_image = Image.open(BytesIO(win_image_bytes))
        
        # Save win image
        win_path = os.path.join(output_dir, "win", f"{i}.png")
        win_image.save(win_path, format="PNG", optimize=True)
        
        # Generate distorted images
        distorted_images = []
        
        for sample_idx in range(total_samples):
            # Randomly select ONE distortion function
            distortion_func = random.choice(available_distortions)
            
            try:
                # Generate distorted image (starts from original_example each time)
                distorted_img = generate_single_distorted_image(
                    original_example, distortion_func, renderer, sample_idx
                )
                
                # Score the distorted image
                score_val = score(distorted_img, original_example['text'])
                distorted_images.append((distorted_img, score_val))
                
            except Exception as e:
                print(f"Error generating sample {sample_idx}: {e}")
                continue
        
        print(f"\nGenerated {len(distorted_images)} distorted images")
        
        # Save checkpoint images
        if i % 200 == 0:
            ckpt_folder = os.path.join(output_dir, f"ckpt_{i}")
            os.makedirs(ckpt_folder, exist_ok=True)
            for idx, (img, _) in enumerate(distorted_images[:20]):  # Save first 20
                img.save(os.path.join(ckpt_folder, f'{idx}.png'))
        
        # Select best subset using DP
        distorted_images = sorted(distorted_images, key=lambda x: x[1])
        
        # Split into buckets
        third = len(distorted_images) // 3
        bucket1 = distorted_images[:third]
        bucket2 = distorted_images[third:2*third]
        bucket3 = distorted_images[2*third:]
        
        # Sample from each bucket
        samples_per_bucket = batch_size // 3
        best_subset = []
        best_subset += max_variation_dp(bucket1, k=samples_per_bucket)
        best_subset += max_variation_dp(bucket2, k=samples_per_bucket)
        best_subset += max_variation_dp(bucket3, k=batch_size - 2*samples_per_bucket)
        
        best_subset = sorted(best_subset, key=lambda x: x[1])
        
        if len(best_subset) < batch_size:
            print(f"Warning: Only got {len(best_subset)} images, skipping...")
            continue
        
        # Save lose images and collect scores
        temp_dict = {}
        lose_paths = []
        
        for j, (img, score_val) in enumerate(best_subset):
            path = os.path.join(output_dir, f"lose{j+1}", f"{i}.png")
            img.save(path)
            lose_paths.append(path)
            temp_dict[j+1] = score_val
        
        # Score win image
        win_score = score(win_image, original_example['text'])
        temp_dict['win_image_score'] = win_score
        
        # Add to dataset
        data_row = {"prompt": original_example['text'], "win_image": win_path}
        for k in range(batch_size):
            data_row[f"lose_image{k+1}"] = lose_paths[k]
        
        final_dataset = pd.concat([final_dataset, pd.DataFrame([data_row])], ignore_index=True)
        json_dict_for_scores.append({i: temp_dict})
        
        print(f"Completed example {i}")
    
    # Save final outputs
    final_csv_path = os.path.join(output_dir, f"final_dataset_{start}_{end}.csv")
    final_json_path = os.path.join(output_dir, f"scores_data_{start}_{end}.json")
    
    final_dataset.to_csv(final_csv_path, index=False)
    with open(final_json_path, 'w') as f:
        json.dump(json_dict_for_scores, f, indent=4)
    
    print(f"\nDataset saved to {final_csv_path}")
    print(f"Scores saved to {final_json_path}")


# ===========================
# USAGE EXAMPLE
# ===========================

if __name__ == "__main__":
    import datasets
    
    # Configuration
    START = 0
    END = 10
    BATCH_SIZE = 16
    TOTAL_SAMPLES = 150
    OUTPUT_DIR = "dataset"
    
    # Load dataset and renderer (you need to implement these)
    dataset = datasets.load_from_disk("my_dataset")
    # renderer = CrelloV5Renderer(dataset.features, fonts_path)  # Your renderer
    
    # Generate dataset
    generate_dataset(START, END, dataset, renderer, BATCH_SIZE, TOTAL_SAMPLES, OUTPUT_DIR)
