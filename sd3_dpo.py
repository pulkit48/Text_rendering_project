"""
DPO Training Script for SD3 Medium — Pre-labeled CSV Dataset
=============================================================
Dataset format (CSV):
    prompt          | winning_image      | losing_image
    "a cat..."      | /path/to/win.jpg   | /path/to/lose.jpg

Key differences from reward-model version:
  - No reward model, no async reward computation
  - No DistributedKRepeatSampler (pairs already labeled)
  - No k-repeat sampling (one pair per prompt row)
  - Images loaded from disk → VAE encoded → latents used directly
  - Much simpler data pipeline overall
"""

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import compute_density_for_timestep_sampling
import numpy as np
import pandas as pd
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.nn.functional as F
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import torchvision.transforms as T
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PreferencePairDataset(Dataset):
    """
    Loads pre-labeled winner/loser image pairs from a CSV file.

    CSV columns (required):
        prompt         — text prompt used to generate both images
        winning_image  — file path to the preferred (chosen) image
        losing_image   — file path to the rejected image

    CSV columns (optional):
        split          — 'train' or 'test' (used if same CSV for both splits)

    Images are resized to (resolution x resolution) and normalized to [-1, 1]
    for VAE encoding.
    """

    def __init__(self, csv_path: str, resolution: int = 512, split: str = None):
        self.resolution = resolution

        df = pd.read_csv(csv_path)

        # ── Validate columns ──────────────────────────────────────────────
        required = {"prompt", "winning_image", "losing_image"}
        missing = required - set(df.columns)
        assert not missing, (
            f"CSV missing columns: {missing}\n"
            f"Found: {list(df.columns)}\n"
            f"Your CSV must have exactly these column names:\n"
            f"  'prompt', 'winning_image', 'losing_image'"
        )

        # ── Optional split filtering ──────────────────────────────────────
        if split is not None:
            assert "split" in df.columns, (
                f"split='{split}' requested but CSV has no 'split' column.\n"
                f"Either:\n"
                f"  (a) Add a 'split' column with values 'train'/'test', OR\n"
                f"  (b) Use separate CSVs: config.train_csv and config.test_csv"
            )
            df = df[df["split"] == split].reset_index(drop=True)
            logger.info(f"Loaded {len(df)} rows for split='{split}'")

        # ── Drop rows with missing image files ────────────────────────────
        before = len(df)
        win_exists = df["winning_image"].apply(os.path.exists)
        lose_exists = df["losing_image"].apply(os.path.exists)
        df = df[win_exists & lose_exists].reset_index(drop=True)
        after = len(df)
        if before != after:
            dropped = before - after
            missing_wins = (~win_exists).sum()
            missing_loses = (~lose_exists).sum()
            logger.warning(
                f"Dropped {dropped} rows with missing image files.\n"
                f"  Missing winning_image files : {missing_wins}\n"
                f"  Missing losing_image files  : {missing_loses}\n"
                f"  Remaining rows              : {after}"
            )

        assert len(df) > 0, (
            f"No valid rows in {csv_path} after filtering.\n"
            f"Check that your image paths are absolute or correct relative paths."
        )

        self.prompts = df["prompt"].tolist()
        self.winning_paths = df["winning_image"].tolist()
        self.losing_paths = df["losing_image"].tolist()

        # Image transform: resize → tensor → normalize to [-1, 1] for VAE
        self.transform = T.Compose([
            T.Resize(
                (resolution, resolution),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.ToTensor(),                        # [0, 1]
            T.Normalize([0.5, 0.5, 0.5],         # → [-1, 1]
                        [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.prompts)

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image at path: '{path}'\n"
                f"Error: {e}\n"
                f"Make sure the file exists and is a valid image (jpg/png/webp)."
            )

    def __getitem__(self, idx: int):
        return {
            "prompt":         self.prompts[idx],
            "winning_image":  self._load_image(self.winning_paths[idx]),
            "losing_image":   self._load_image(self.losing_paths[idx]),
            "winning_path":   self.winning_paths[idx],
            "losing_path":    self.losing_paths[idx],
        }

    @staticmethod
    def collate_fn(examples):
        """
        Returns:
            prompts        : list[str], length B
            winning_images : Tensor [B, 3, H, W]  in [-1, 1]
            losing_images  : Tensor [B, 3, H, W]  in [-1, 1]
            metadata       : list[dict] with 'winning_path', 'losing_path'
        """
        prompts        = [e["prompt"]        for e in examples]
        winning_images = torch.stack([e["winning_image"] for e in examples])
        losing_images  = torch.stack([e["losing_image"]  for e in examples])
        metadata       = [
            {"winning_path": e["winning_path"], "losing_path": e["losing_path"]}
            for e in examples
        ]
        return prompts, winning_images, losing_images, metadata


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_text_embeddings(prompts, text_encoders, tokenizers,
                            max_sequence_length, device):
    """
    Encode prompts with all available encoders.
    Handles T5 being None (drop_t5) or on CPU (cpu_offload_t5).
    """
    with torch.no_grad():
        t5 = text_encoders[2]
        offload_t5 = (
            t5 is not None
            and next(t5.parameters()).device.type == "cpu"
        )
        if offload_t5:
            t5.to(device)

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompts, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        if offload_t5:
            t5.to("cpu")
            torch.cuda.empty_cache()

    return prompt_embeds, pooled_prompt_embeds


@torch.no_grad()
def encode_images_to_latents(images: torch.Tensor, vae,
                              device, dtype=torch.float32) -> torch.Tensor:
    """
    Encode pixel images → VAE latents.

    Args:
        images : [B, 3, H, W] normalized to [-1, 1]
    Returns:
        latents : [B, 16, H//8, W//8]

    VAE is kept in float32. SD3 uses shift_factor + scaling_factor.
    """
    images = images.to(device=device, dtype=torch.float32)
    latents = vae.encode(images).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents.to(dtype=dtype)


def get_sigmas(noise_scheduler, timesteps, accelerator,
               n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [
        (schedule_timesteps == t).nonzero().item() for t in timesteps
    ]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def copy_learner_to_ref(transformer):
    param_dict = dict(transformer.named_parameters())
    for name, param in param_dict.items():
        if "learner" in name:
            ref_name = name.replace("learner", "ref")
            if ref_name in param_dict:
                param_dict[ref_name].data.copy_(param.data)


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(save_dir, transformer, global_step, accelerator,
              ema, transformer_trainable_parameters, config):
    save_root = os.path.join(
        save_dir, "checkpoints", f"checkpoint-{global_step}"
    )
    os.makedirs(os.path.join(save_root, "lora"), exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(
            transformer, accelerator
        ).base_model.model.save_pretrained(
            os.path.join(save_root, "lora")
        )
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)



# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = (
        unique_id if not config.run_name else config.run_name + "_" + unique_id
    )

    num_train_timesteps = int(
        config.sample.num_steps * config.train.timestep_fraction
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=os.path.join(config.logdir, config.run_name),
            automatic_checkpoint_naming=True,
            total_limit=config.num_checkpoint_limit,
        ),
        gradient_accumulation_steps=(
            config.train.gradient_accumulation_steps * num_train_timesteps
        ),
    )

    if accelerator.is_main_process:
        wandb.init(project="dpo_sd3_medium_csv", name=config.run_name)

    logger.info(f"\n{config}")
    set_seed(config.seed, device_specific=True)

    # ── Pipeline ───────────────────────────────────────────────────────────
    drop_t5       = getattr(config, 'drop_t5_encoder',    False)
    cpu_offload_t5 = getattr(config, 'cpu_offload_t5',   False)

    load_kwargs = {}
    if drop_t5:
        load_kwargs = {"text_encoder_3": None, "tokenizer_3": None}
        logger.info("T5 encoder disabled — ~9 GB VRAM saved.")

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model, **load_kwargs
    )

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    if pipeline.text_encoder_3 is not None:
        pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    pipeline.safety_checker = None

    text_encoders = [
        pipeline.text_encoder,
        pipeline.text_encoder_2,
        pipeline.text_encoder_3,   # may be None
    ]
    tokenizers = [
        pipeline.tokenizer,
        pipeline.tokenizer_2,
        pipeline.tokenizer_3,      # may be None
    ]

    pipeline.set_progress_bar_config(
        position=1, disable=not accelerator.is_local_main_process,
        leave=False, desc="Timestep", dynamic_ncols=True,
    )

    # ── Dtype ──────────────────────────────────────────────────────────────
    inference_dtype = {
        "fp16": torch.float16, "bf16": torch.bfloat16
    }.get(accelerator.mixed_precision, torch.float32)

    pipeline.vae.to(accelerator.device, dtype=torch.float32)  # VAE stays fp32
    pipeline.text_encoder.to(accelerator.device,  dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    if pipeline.text_encoder_3 is not None:
        target_device = "cpu" if cpu_offload_t5 else accelerator.device
        pipeline.text_encoder_3.to(target_device, dtype=inference_dtype)
        if cpu_offload_t5:
            logger.info("T5 encoder on CPU — will be offloaded to GPU during encoding.")
    pipeline.transformer.to(accelerator.device)

    # ── LoRA ───────────────────────────────────────────────────────────────
    if config.use_lora:
        lora_cfg = LoraConfig(
            r=32, lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=[
                "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
                "attn.to_add_out", "attn.to_k", "attn.to_out.0",
                "attn.to_q", "attn.to_v",
            ],
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, config.train.lora_path
            )
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(
                pipeline.transformer, lora_cfg, adapter_name="learner"
            )
            pipeline.transformer = get_peft_model(
                pipeline.transformer, lora_cfg, adapter_name="ref"
            )
            pipeline.transformer.set_adapter("learner")
            # ===== FIX: initialize ref = learner =====
            copy_learner_to_ref(pipeline.transformer)

            # ===== memory optimization (safe) =====
            if hasattr(pipeline.transformer, "enable_xformers_memory_efficient_attention"):
                pipeline.transformer.enable_xformers_memory_efficient_attention()

    if getattr(config, 'gradient_checkpointing', True):
        pipeline.transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled.")

    transformer = pipeline.transformer
    transformer_trainable_parameters = [
        p for n, p in transformer.named_parameters() if "learner" in n
    ]

    ema = EMAModuleWrapper(
        transformer_trainable_parameters, decay=0.9,
        update_step_interval=8, device=accelerator.device,
    )

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Optimizer ──────────────────────────────────────────────────────────
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("8-bit AdamW enabled.")
        except ImportError:
            raise ImportError("pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # ── Dataset — CSV-based ────────────────────────────────────────────────
    #
    # Option A — single CSV with a 'split' column:
    #   config.dataset = "/data/preferences.csv"   (has 'split' col)
    #
    # Option B — separate train/test CSVs:
    #   config.train_csv = "/data/train.csv"
    #   config.test_csv  = "/data/test.csv"
    #
    train_csv = getattr(config, 'train_csv', None) or config.dataset
    test_csv  = getattr(config, 'test_csv',  None) or config.dataset
    use_split_col = (train_csv == test_csv)

    train_dataset = PreferencePairDataset(
        csv_path=train_csv,
        resolution=config.resolution,
        split='train' if use_split_col else None,
    )
    test_dataset = PreferencePairDataset(
        csv_path=test_csv,
        resolution=config.resolution,
        split='test' if use_split_col else None,
    )

    logger.info(f"Train pairs : {len(train_dataset)}")
    logger.info(f"Test pairs  : {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=PreferencePairDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=PreferencePairDataset.collate_fn,
        pin_memory=True,
    )

    autocast = accelerator.autocast

    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader, test_dataloader
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained.model, subfolder="scheduler"
    )

    logger.info("***** DPO — SD3 Medium — CSV dataset *****")
    logger.info(f"  Resolution     = {config.resolution}")
    logger.info(f"  Batch size     = {config.train.batch_size}")
    logger.info(f"  T5 dropped     = {drop_t5}")
    logger.info(f"  T5 CPU offload = {cpu_offload_t5}")

    epoch = 0
    global_step = 0

    while True:


        # ── TRAINING ──────────────────────────────────────────────────────
        pipeline.transformer.set_adapter("learner")
        pipeline.transformer.train()
        info = defaultdict(list)

        for batch in tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            prompts, winning_images, losing_images, metadata = batch
            # winning_images / losing_images : [B, 3, H, W] in [-1, 1]

            # Periodic ref sync
            if global_step > 0 and global_step % config.train.ref_update_step == 0:
                copy_learner_to_ref(transformer)

            # ── Encode images → latents (outside autocast, VAE in fp32) ──
            with torch.no_grad():
                winning_latents = encode_images_to_latents(
                    winning_images, pipeline.vae,
                    device=accelerator.device, dtype=inference_dtype,
                )
                losing_latents = encode_images_to_latents(
                    losing_images, pipeline.vae,
                    device=accelerator.device, dtype=inference_dtype,
                )

            # [2B, C, H, W] — first bsz = chosen, last bsz = rejected
            bsz = winning_latents.shape[0]
            model_input = torch.cat([winning_latents, losing_latents], dim=0)

            # ── Text embeddings ───────────────────────────────────────────
            with torch.no_grad():
                pe, ppe = compute_text_embeddings(
                    prompts, text_encoders, tokenizers,
                    max_sequence_length=128, device=accelerator.device,
                )
            # Repeat for chosen + rejected (same prompt)
            pe  = torch.cat([pe,  pe],  dim=0)
            ppe = torch.cat([ppe, ppe], dim=0)

            # ── Timestep loop ─────────────────────────────────────────────
            for j in range(num_train_timesteps):
                with accelerator.accumulate(transformer):

                    # Shared noise — Diffusion-DPO requirement
                    noise = torch.randn_like(model_input)
                    noise = torch.cat([noise[:bsz], noise[:bsz]], dim=0)

                    # Logit-normal timestep sampling
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme='logit_normal',
                        batch_size=bsz,
                        logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                    timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)
                    timesteps = torch.cat([timesteps, timesteps], dim=0)

                    # x_t = (1 - σ) * x₀  +  σ * noise
                    sigmas = get_sigmas(
                        noise_scheduler, timesteps, accelerator,
                        n_dim=model_input.ndim, dtype=model_input.dtype,
                    )
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    with autocast():
                        # Learner
                        pipeline.transformer.set_adapter("learner")
                        model_pred = transformer(
                            hidden_states=noisy_model_input,
                            timestep=timesteps,
                            encoder_hidden_states=pe,
                            pooled_projections=ppe,
                            return_dict=False,
                        )[0]

                        # Ref (frozen)
                        with torch.no_grad():
                            pipeline.transformer.set_adapter("ref")
                            model_pred_ref = transformer(
                                hidden_states=noisy_model_input,
                                timestep=timesteps,
                                encoder_hidden_states=pe,
                                pooled_projections=ppe,
                                return_dict=False,
                            )[0].detach()
                            pipeline.transformer.set_adapter("learner")

                    # Flow-matching target: v = noise - x₀
                    target = noise - model_input

                    # Per-sample MSE
                    theta_mse = (
                        (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1).mean(dim=1)
                    ref_mse = (
                        (model_pred_ref.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1).mean(dim=1)

                    model_w_err = theta_mse[:bsz]
                    model_l_err = theta_mse[bsz:]
                    ref_w_err   = ref_mse[:bsz]
                    ref_l_err   = ref_mse[bsz:]

                    w_diff   = model_w_err - ref_w_err
                    l_diff   = model_l_err - ref_l_err
                    w_l_diff = w_diff - l_diff

                    inside_term = -0.5 * config.train.beta * w_l_diff
                    loss = torch.mean(-F.logsigmoid(inside_term))

                    info["loss"].append(loss.detach())
                    info["model_w_err"].append(model_w_err.mean().detach())
                    info["model_l_err"].append(model_l_err.mean().detach())
                    info["ref_w_err"].append(ref_w_err.mean().detach())
                    info["ref_l_err"].append(ref_l_err.mean().detach())
                    info["w_diff"].append(w_diff.mean().detach())
                    info["l_diff"].append(l_diff.mean().detach())
                    info["w_l_diff"].append(w_l_diff.mean().detach())
                    info["inside_term"].append(inside_term.mean().detach())
                    info["implicit_acc"].append(
                        (inside_term > 0).float().mean().detach()
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            transformer.parameters(),
                            config.train.max_grad_norm,
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    logged = {
                        k: torch.mean(torch.stack(v)) for k, v in info.items()
                    }
                    logged = accelerator.reduce(logged, reduction="mean")
                    logged["epoch"] = epoch
                    if accelerator.is_main_process:
                        wandb.log(logged, step=global_step)
                    info = defaultdict(list)
                    global_step += 1
                    if config.train.ema:
                        ema.step(transformer_trainable_parameters, global_step)
                
                    # ===== SAVE EVERY 50 STEPS =====
                    if accelerator.is_main_process and global_step % 50 == 0 and global_step > 0:
                        save_ckpt(
                            config.save_dir,
                            transformer,
                            global_step,
                            accelerator,
                            ema,
                            transformer_trainable_parameters,
                            config,
                        )

        epoch += 1


if __name__ == "__main__":
    app.run(main)
