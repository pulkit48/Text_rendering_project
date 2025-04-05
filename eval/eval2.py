from metric.clip import calculate_clip_score
from metric.str_match import exact_match_accuracy, exact_match_accuracy_case_insensitive,normalized_edit_distance, longest_common_subsequence_length, normalized_longest_common_subsequence
from metric.human_pref_metric import hpsv2_score
import pandas as pd
import numpy as np
import cv2
import easyocr
import warnings
import torch
from PIL import Image
warnings.filterwarnings("ignore")

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = easyocr.Reader(['en'],gpu=True)
    print("Hi, I am here")
    
    root_folder="eval"                      # Root folder where everything is present
    dataset_name=args.data.split(".")[0]    # Fetching the dataset name
    result_dir=f"{root_folder}/results/{args.model}/{dataset_name}"   # Dir where result will get saved
    result_file_path=f"{result_dir}/result.csv"   # Path to the generated result
    score_file_path=f"{result_dir}/scores.csv"   # Path to the generated scores

    df=pd.read_csv(result_file_path)
    df['ocr_text'] = ""  
    clip_avg=0
    em_avg=0
    em_ci_avg=0
    ned_avg=0
    nlcs_avg=0
    image_reward_avg=0
    hpsv2_avg=0

    print("Hi, I am here2")
    for i in range(len(df)):
        # print(i)
        prompt=df.iloc[i]['prompt']
        img_path=df.iloc[i]['img_path']
        
        
        match = re.search(r'"(.*?)"', text)
        if match:
            prompt_main_text = match.group(1)
        else:
            prompt_main_text = ""

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ocr_text = reader.readtext(img_rgb, detail=0)
        df['ocr_text'][i] = " ".join(ocr_text)

        
        clip_val=calculate_clip_score(prompt, img_path)
        em=exact_match_accuracy(prompt_main_text, ocr_text)
        em_ci=exact_match_accuracy_case_insensitive(prompt_main_text, ocr_text)
        ned=normalized_edit_distance(prompt_main_text, ocr_text)
        lcs=normalized_longest_common_subsequence(prompt_main_text, ocr_text)
        # image_reward_val=image_reward(prompt, img_path)
        hpsv2_val=hpsv2_score(prompt, img_path)

        clip_avg+=clip_val
        clip_avg/=(i+1)
        em_avg+=em
        em_avg/=(i+1)
        em_ci_avg+=em_ci
        em_ci_avg/=(i+1)
        ned_avg+=ned
        ned_avg/=(i+1)
        nlcs_avg+=lcs
        nlcs_avg/=(i+1)
        # image_reward_avg+=image_reward_val
        # image_reward_avg/=(i+1)
        hpsv2_avg+=hpsv2_val
        hpsv2_avg/=(i+1)

        print(f"Image {i+1}/{len(df)}: CLIP Score: {clip_val}, EM: {em}, EM (CI): {em_ci}, NED: {ned}, NLCS: {lcs}, HPSv2: {hpsv2_val}")
    
    df.to_csv(result_file_path, index=False)
    
    print(f"Average CLIP Score: {clip_avg}")
    print(f"Average Exact Match Accuracy: {em_avg}")
    print(f"Average Exact Match Accuracy (Case Insensitive): {em_ci_avg}")
    print(f"Average Normalized Edit Distance: {ned_avg}")
    print(f"Average Normalized Longest Common Subsequence: {nlcs_avg}")
    # print(f"Average Image Reward: {image_reward_avg}")
    print(f"Average HPSv2: {hpsv2_avg}")

    score_dict={
        "Average CLIP Score": clip_avg,
        "Average Exact Match Accuracy": em_avg,
        "Average Exact Match Accuracy (Case Insensitive)": em_ci_avg,
        "Average Normalized Edit Distance": ned_avg,
        "Average Normalized Longest Common Subsequence": nlcs_avg,
        # "Average Image Reward": image_reward_avg,
        "Average HPSv2": hpsv2_avg
    }

    print(score_dict)
    
    with open(score_file_path, "w") as f:
        f.write("Metric,Score\n")
        for key, value in score_dict.items():
            f.write(f"{key},{value}\n")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the model's performance")
    parser.add_argument("--model", type=str, default="kolors", choices=["kolors", "sd3", "flux","sdxl"], help="Model to use for evaluation")
    parser.add_argument("--data", type=str, default="ours.txt",choices=["ours.txt", "anytext.txt","creativebench.txt","mario.txt"], help="Path to data file")
    args = parser.parse_args()
    
    main(args)
        