from huggingface_hub import login
import torch
import os
import warnings
import pandas as pd
import numpy as np
import cv2
warnings.filterwarnings("ignore")

# Log in with Hugging Face token
login(token="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")

def main(args):

    # Chnage this path as per your requirement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    root_folder="eval"                                                # Root folder where everything is present
    dataset_name=args.data.split(".")[0]                              # Fetching the dataset name
    result_dir=f"{root_folder}/results/{args.model}/{dataset_name}"   # Dir where result will get saved
    data_file_path=f"{root_folder}/eval_data/{args.data}"             # Path to the dataset
    os.makedirs(f"{result_dir}/images", exist_ok=True)

    
    with open(data_file_path, "r") as f:
        data = f.readlines()
        data = [line.strip() for line in data if line.strip()]

    prompt_list=[]
    img_path_list=[]
    
    
    # Load the model based on the argument
    if args.model == "kolors":
        from diffusers import KolorsPipeline
        pipe = KolorsPipeline.from_pretrained("Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16").to(device)
        
    elif args.model == "sd3":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16).to(device)

    elif args.model == "flux":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    
    elif args.model == "sdxl":
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to(device)

    
    # Generate images
    for ind, text in enumerate(data):
        if ind%10==0:
            print(f"Processing {ind}/{len(data)}")

        text = text.strip()
        if dataset_name == "ours":
            prompt = f'Generate an image with text "{text}" written on it.'
        else:
            prompt=text
        img_path = f"{result_dir}/images/img_{ind}.png"
        if os.path.exists(img_path):
            continue
        if args.model == "kolors":
            image = pipe(prompt=prompt, negative_prompt="", height=512, width=512, guidance_scale=5.0, num_inference_steps=50, generator=torch.Generator(pipe.device).manual_seed(66)).images[0]
        elif args.model == "sd3":
            image = pipe(prompt, num_inference_steps=28,height=512, width=512, guidance_scale=3.5, generator=torch.Generator("cuda").manual_seed(0)).images[0]
        elif args.model == "flux":
            image = pipe(prompt, height=512, width=512, guidance_scale=3.5, num_inference_steps=50, max_sequence_length=512, generator=torch.Generator("cuda").manual_seed(0)).images[0]
        elif args.model=='sdxl':
            image = pipe(prompt, height=512, width=512, guidance_scale=7.5, num_inference_steps=50).images[0]

        image.save(img_path)

        

        prompt_list.append(prompt)
        img_path_list.append(img_path)
        
    
    df=pd.DataFrame({"prompt": prompt_list, "img_path": img_path_list})
    df.to_csv(f"{result_dir}/result.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="kolors", choices=["kolors", "sd3", "flux","sdxl"], help="Model to use for image generation")
    parser.add_argument("--data", type=str, default="ours.txt",choices=["ours.txt", "anytext.txt","creativebench.txt","mario.txt"], help="Path to data file")
    args = parser.parse_args()
    main(args)