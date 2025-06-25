export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATASET_NAME="data.csv"

accelerate launch --num_processes=1 working_code.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir="diffusion_dspo_sd1.5" \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --learning_rate=1e-8 --scale_lr \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=500 \
  --num_train_epochs=100 \
  --sft=True \
  --checkpointing_steps=50 \
  --beta_dpo 0.001 \
