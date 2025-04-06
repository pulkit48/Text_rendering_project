from metric.clip import calculate_clip_score
from metric.aesthetic_score import calculate_aesthetic_score
from metric.human_pref_metric import hpsv2_score
import pandas as pd
import numpy as np
import cv2
import easyocr
import re
import warnings
import torch
from PIL import Image
warnings.filterwarnings("ignore")

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = easyocr.Reader(['en'],gpu=True)
    print("Hi, I am here")
    
    root_folder="eval_image_quality"                      # Root folder where everything is present
    dataset_name=args.data.split(".")[0]    # Fetching the dataset name
    result_dir=f"{root_folder}/results/{args.model}/{dataset_name}"   # Dir where result will get saved
    result_file_path=f"{result_dir}/result.csv"   # Path to the generated result
    score_file_path=f"{result_dir}/scores.csv"   # Path to the generated scores

    df=pd.read_csv(result_file_path)
    df['ocr_text'] = ""  
    clip_avg=0
    asthetic_avg=0
    hpsv2_avg=0

    print("Hi, I am here2")
    for i in range(len(df)):
        # print(i)
        prompt=df.iloc[i]['prompt']
        img_path=df.iloc[i]['img_path']
        

        clip_val=calculate_clip_score(prompt, img_path)
        asthetic_val=calculate_aesthetic_score(img_path)
        hpsv2_val=hpsv2_score(prompt, img_path)

        clip_avg+=clip_val
        asthetic_avg+=asthetic_val
        hpsv2_avg+=hpsv2_val

        print(f"Image {i+1}/{len(df)}: CLIP Score: {clip_avg/(i+1)}, Aesthetic Score: {asthetic_avg/(i+1)}, HPSv2: {hpsv2_avg/(i+1)}")
    
    # df.to_csv(result_file_path, index=False)
    clip_avg /= len(df)
    asthetic_avg /= len(df)
    hpsv2_avg /= len(df)
    print(f"Average CLIP Score: {clip_avg}")
    print(f"Average Aesthetic Score: {asthetic_avg}")
    # print(f"Average Image Reward: {image_reward_avg}")
    print(f"Average HPSv2: {hpsv2_avg}")

    score_dict={
        "Average CLIP Score": clip_avg,
        "Average Aesthetic Score": asthetic_avg,
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
    parser.add_argument("--model", type=str, default="kolors", choices=["kolors", "sd3", "flux","sdxl","sdxl-dpo"], help="Model to use for evaluation")
    parser.add_argument("--data", type=str, default="drawbench.txt",choices=["drawbench.txt","hpsv2.txt","partiprompt.txt","picapic.txt"], help="Path to data file")
    args = parser.parse_args()
    
    main(args)
        