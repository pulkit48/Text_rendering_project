export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="data.csv"

accelerate launch --num_processes=1 working_code.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --output_dir="diffusion_dspo_sdxl" \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --num_train_epochs=100 \
  --checkpointing_steps 100 \
  --beta_dpo 5000 \
  --sdxl \
