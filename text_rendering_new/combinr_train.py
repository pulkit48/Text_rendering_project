import argparse
import io
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, DistributedType
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel,     StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

## SDXL
import functools
import gc
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer, PretrainedConfig

from peft import LoraConfig, set_peft_model_state_dict
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin, StableDiffusionLoraLoaderMixin
from diffusers.training_utils import cast_training_params
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers,convert_unet_state_dict_to_peft


logger = get_logger(__name__, log_level="INFO")


def get_adaptive_lose_l_scale(pred_eps: torch.Tensor,
                              true_eps: torch.Tensor,
                              mu: float,
                              eps: float = 1e-9,
                              max_lambda: float = 1.0) -> torch.Tensor:
    """
    Returns λ based on the winner-preserving condition (used only to scale the gradient of the lose branch).
    Constraint: (-∇A + λ∇B) · ∇A ≤ -μ ||∇A||^2  ⇒  when ∇B · ∇A > 0,
            λ ≤ ((1 - μ) ||∇A||^2) / (∇B · ∇A)

    pred_eps/true_eps are assumed to be arranged along the batch dimension: the first half is win, the second half is lose.
    """
    assert 0.0 <= mu <= 1.0, "mu should be within [0,1]"

    # Split into win/lose along the batch dimension
    pred_w, pred_l = pred_eps.detach().requires_grad_(True).chunk(2, dim=0)
    true_w, true_l = true_eps.detach().chunk(2, dim=0)

    # Use the noise-prediction MSE as a scalar proxy for A and B
    A = ((pred_w - true_w) ** 2).mean()
    B = ((pred_l - true_l) ** 2).mean()

    # Output-space proxy gradient
    gA = torch.autograd.grad(A, pred_w, create_graph=False, retain_graph=False)[0]
    gB = torch.autograd.grad(B, pred_l, create_graph=False, retain_graph=False)[0]

    # Scalars: ||gA||^2 and gB·gA
    num = (gA.flatten().pow(2).sum())                     # ||gA||^2
    den = (gB.flatten() * gA.flatten()).sum()             # gB·gA

    # When den > 0 (the two gradients point in the same direction), apply the upper bound; otherwise, leave unconstrained (assign a large value)
    lam_cap = ((1.0 - float(mu)) * num) / (den + eps)
    lam = torch.where(
        den > 0,
        lam_cap,
        torch.tensor(max_lambda, device=pred_eps.device, dtype=pred_eps.dtype)
    )

    # Numerical safeguard: ensure non-negativity and apply clipping
    lam = lam.clamp(min=0.0, max=max_lambda).detach().to(dtype=pred_eps.dtype)
    return lam
        
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="prompt",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        # was random for submission, need to test that not distributing same noise etc across devices
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "If set the images will be randomly"
            " cropped (instead of center). The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_adafactor", action="store_true", help="Whether or not to use adafactor (should save mem)"
    )
    # Bram Note: Haven't looked @ this yet
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tuning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    ## SDXL
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument("--sdxl", action='store_true', help="Train sdxl")
    
    ## DPO
    parser.add_argument("--sft", action='store_true', help="Run Supervised Fine-Tuning instead of Direct Preference Optimization")
    parser.add_argument("--beta_dpo", type=float, default=5000, help="The beta DPO temperature controlling strength of KL penalty")
    parser.add_argument(
        "--hard_skip_resume", action="store_true", help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever"
    )
    parser.add_argument(
        "--unet_init", type=str, default='', help="Initialize start of run from unet (not compatible w/ checkpoint load)"
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.2,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--split", type=str, default='train', help="Datasplit"
    )
    parser.add_argument(
        "--choice_model", type=str, default='', help="Model to use for ranking (override dataset PS label_0/1). choices: aes, clip, hps, pickscore"
    )
    
    parser.add_argument(
        "--use_winner_preserving", action="store_true"
    )
    parser.add_argument(
        "--winner_preserving_mu", type=float, default=0.0
    )
    parser.add_argument(
        "--dmpo_alpha", type=float, default=0.0
    )
    parser.add_argument(
        "--train_method", type=str, default='diffusion-dpo', choices=['diffusion-dpo', 'dspo', 'dmpo', 'sft']
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    

    ## SDXL
    if args.sdxl:
        print("Running SDXL")
    if args.resolution is None:
        if args.sdxl:
            args.resolution = 1024
        else:
            args.resolution = 512
            
    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_sdxl(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True,device="cuda"):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}




def main():
    
    args = parse_args()
    
    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index) # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ### END ACCELERATOR BOILERPLATE
    

    def enforce_zero_terminal_snr(scheduler):
        
        alphas = 1 - scheduler.betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
    
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        scheduler.alphas_cumprod = alphas_cumprod
        return 

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, 
                                                    subfolder="scheduler")

    if 'turbo' in args.pretrained_model_name_or_path:
        enforce_zero_terminal_snr(noise_scheduler)
    

    if args.sdxl:
        # Load the tokenizers
        if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
            tokenizer_and_encoder_name = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            tokenizer_and_encoder_name = args.pretrained_model_name_or_path
        tokenizer_one = AutoTokenizer.from_pretrained(
            tokenizer_and_encoder_name, subfolder="tokenizer", revision=args.revision, use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )

    
    if args.sdxl:
        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            tokenizer_and_encoder_name, args.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            tokenizer_and_encoder_name, args.revision, subfolder="text_encoder_2"
        )
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            tokenizer_and_encoder_name, subfolder="text_encoder", revision=args.revision
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
        )
        if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
            text_encoders = [text_encoder_two]
            tokenizers = [tokenizer_two]
        else:
            text_encoders = [text_encoder_one, text_encoder_two]
            tokenizers = [tokenizer_one, tokenizer_two]
    else:
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
    # Can custom-select VAE (used in original SDXL tuning)

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path, subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None, revision=args.revision
    )

    if args.unet_init:
        print("Initializing unet from", args.unet_init)
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_init if args.unet_init else args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )

    unet.add_adapter(unet_lora_config)

    if args.mixed_precision == "fp16":
        for param in unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

    if args.sdxl:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                weights.pop()

            # ✅ pick correct mixin
            lora_mixin = StableDiffusionXLLoraLoaderMixin if args.sdxl else StableDiffusionLoraLoaderMixin
            lora_mixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
            )
    
    def load_model_hook(models, input_dir):
        unet_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        # ✅ pick correct mixin
        lora_mixin = StableDiffusionXLLoraLoaderMixin if args.sdxl else StableDiffusionLoraLoaderMixin
        lora_state_dict, network_alphas = lora_mixin.lora_state_dict(input_dir)

        unet_state_dict = {
            f"{k.replace('unet.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")

        if args.mixed_precision == "fp16":
            cast_training_params([unet_])


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing or args.sdxl: #  (args.sdxl and ('turbo' not in args.pretrained_model_name_or_path) ):
        print("Enabling gradient checkpointing, either because you asked for this or because you're using SDXL")
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

    if args.use_adafactor or args.sdxl:
        print("Using Adafactor either because you asked for it or you're using SDXL")
        optimizer = transformers.Adafactor(params_to_optimize,
                                           lr=args.learning_rate,
                                           weight_decay=args.adam_weight_decay,
                                           clip_threshold=1.0,
                                           scale_parameter=False,
                                          relative_step=False)
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )


    dataset = load_dataset(
    'csv',
    data_files={args.split: args.dataset_name},  # ✅ maps your file to the split key
    cache_dir=args.cache_dir,
    )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[args.caption_column]:  # ← uses args.caption_column
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{args.caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    if args.train_method != 'sft':
        def preprocess_train(examples):
            all_pixel_values = []
            for col_name in ['win', 'lose']:  # ← your CSV columns
                images = [Image.open(path).convert("RGB")  # ← open from file path, not bytes
                        for path in examples[col_name]]
                pixel_values = [train_transforms(image) for image in images]
                all_pixel_values.append(pixel_values)

            # No label flipping needed — win is always first in your CSV
            combined_pixel_values = []
            for im_tup in zip(*all_pixel_values):
                combined_im = torch.cat(im_tup, dim=0)  # shape: [6, H, W]
                combined_pixel_values.append(combined_im)

            examples["pixel_values"] = combined_pixel_values
            if not args.sdxl:
                examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d = {"pixel_values": pixel_values}
            if args.sdxl:
                return_d[args.caption_column] = [example[args.caption_column] for example in examples]  # ← args.caption_column
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            return return_d

    elif args.train_method == 'sft':
        def preprocess_train(examples):
            # Picks the winning image for SFT (supervised fine-tuning on preferred images only)
            images = [Image.open(path).convert("RGB") for path in examples['win']]  # ← always use win for sft
            examples["pixel_values"] = [train_transforms(image) for image in images]
            if not args.sdxl:
                examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d = {"pixel_values": pixel_values}
            if args.sdxl:
                return_d[args.caption_column] = [example[args.caption_column] for example in examples]  # ← args.caption_column
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            return return_d

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset[args.split] = dataset[args.split].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset[args.split].with_transform(preprocess_train)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(args.split=='train'),
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    print('current weight dtype is {}'.format(weight_dtype))

    vae.to(accelerator.device, dtype=weight_dtype)
    if args.sdxl:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Training initialization
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Training Method = {args.train_method}")
    global_step = 0
    first_epoch = 0
    resume_step = 0     # ← add this line

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

        
    #### START MAIN TRAINING LOOP #####
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step and (not args.hard_skip_resume):
                if step % args.gradient_accumulation_steps == 0:
                    print(f"Dummy processing step {step}, will start training at {resume_step}")
                continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.train_method != 'sft':
                    feed_pixel_values = torch.cat(batch["pixel_values"].chunk(2, dim=1))

                    
                elif args.train_method == 'sft':
                    feed_pixel_values = batch["pixel_values"]

                with torch.no_grad():
                    latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                if args.noise_offset: # haven't tried yet
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                if 'refiner' in args.pretrained_model_name_or_path:
                    timesteps = timesteps % 200
                elif 'turbo' in args.pretrained_model_name_or_path:
                    timesteps_0_to_3 = timesteps % 4
                    timesteps = 250 * timesteps_0_to_3 + 249
                
                if args.train_method != 'sft': # make timesteps and noise same for pairs in DPO
                    timesteps = timesteps.chunk(2)[0].repeat(2)
                    noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                noisy_latents = noise_scheduler.add_noise(latents,
                                                          noise,
                                                          timesteps)
                ### START PREP BATCH ###
                if args.sdxl:
                    # Get the text embedding for conditioning
                    with torch.no_grad():

                        if 'refiner' in args.pretrained_model_name_or_path:
                            add_time_ids = torch.tensor([args.resolution, 
                                                         args.resolution,
                                                         0,
                                                         0,
                                                          6.0], # aesthetics conditioning https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L691C9-L691C24
                                                         dtype=weight_dtype,
                                                         device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                        else: # SDXL-base
                            add_time_ids = torch.tensor([args.resolution, 
                                                         args.resolution,
                                                         0,
                                                         0,
                                                          args.resolution, 
                                                         args.resolution],
                                                         dtype=weight_dtype,
                                                         device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                        prompt_batch = encode_prompt_sdxl(batch, 
                                                          text_encoders,
                                                           tokenizers,
                                                           args.proportion_empty_prompts, 
                                                          args.caption_column,
                                                           is_train=True,
                                                           device=accelerator.device,
                                                          )
                    if args.train_method != 'sft':
                        prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(2, 1, 1)
                        prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_prompt_embeds"].repeat(2, 1)
                    unet_added_conditions = {"time_ids": add_time_ids,
                                            "text_embeds": prompt_batch["pooled_prompt_embeds"]}
                else: # sd1.5
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    if args.train_method != 'sft':
                        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
                #### END PREP BATCH ####
                        
                assert noise_scheduler.config.prediction_type == "epsilon"
                target = noise
               
                model_batch_args = (noisy_latents,
                                    timesteps, 
                                    prompt_batch["prompt_embeds"] if args.sdxl else encoder_hidden_states)
                added_cond_kwargs = unet_added_conditions if args.sdxl else None
                
                model_pred = unet(
                                *model_batch_args,
                                  added_cond_kwargs = added_cond_kwargs
                                 ).sample
                #### START LOSS COMPUTATION ####
                if args.train_method == 'sft': # SFT, casting for F.mse_loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
                    # losses are both 2 * LBS
                    # 1st half of tensors is preferred (y_w), second half is unpreferred
                    model_losses = (model_pred - target).pow(2).mean(dim=[1,2,3])
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    # below for logging purposes
                    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())

                    if args.use_winner_preserving:
                        lose_l_scale = get_adaptive_lose_l_scale(model_pred, target, args.winner_preserving_mu)
                    else:
                        lose_l_scale = 1.0

                    model_losses_l_scaled = model_losses_l.detach() + lose_l_scale * (model_losses_l - model_losses_l.detach())
                    model_diff = model_losses_w - model_losses_l_scaled
                    
                    with torch.no_grad(): # Get the reference policy (unet) prediction
                        accelerator.unwrap_model(unet).disable_adapters()
                        ref_pred = unet(
                                    *model_batch_args,
                                      added_cond_kwargs = added_cond_kwargs
                                     ).sample.detach()
                        accelerator.unwrap_model(unet).enable_adapters()
                        ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                        ref_diff = ref_losses_w - ref_losses_l
                        raw_ref_loss = ref_losses.mean()    
                        
                    scale_term = -0.5 * args.beta_dpo
                    inside_term = scale_term * (model_diff - ref_diff)
                    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                    if args.train_method == 'diffusion-dpo':
                        loss = -1 * F.logsigmoid(inside_term).mean()
                    elif args.train_method == 'dspo':
                        pred2, _ = (model_pred - ref_pred).chunk(2)
                        if args.sdxl:
                            if pred2.pow(2).mean(dim=[1,2,3]).mean() >= 0.1:
                                loss = ((1 - F.sigmoid(inside_term)[:, None, None, None]) * pred2).pow(2).mean(dim=[1,2,3]).mean()
                            else:
                                loss =  (1 - F.sigmoid(inside_term)[:, None, None, None]).pow(2).mean(dim=[1,2,3]).mean()
                        else:
                            model_diff_w, _ = (model_pred - target).chunk(2)
                            loss = (model_diff_w - args.beta_dpo * (1 - F.sigmoid(inside_term)[:, None, None, None]) * pred2).pow(2).mean(dim=[1,2,3]).mean()
                    elif args.train_method == 'dmpo':
                        ut = inside_term
                        p = torch.sigmoid(ut)
                        
                        eps = 1e-6
                        q = p.new(data=(1.0 - args.dmpo_alpha,)).reshape(-1)
                        q = q.clamp(eps, 1 - eps)
                        

                        # log p = -softplus(-u), log(1-p) = -softplus(u)
                        log_p = -F.softplus(-ut)
                        log_1mp = -F.softplus(ut)

                        # KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
                        loss = p * (log_p - torch.log(q)) + (1 - p) * (log_1mp - torch.log1p(-q))
                        loss = loss.mean()
                    else:
                        raise NotImplementedError
                #### END LOSS COMPUTATION ###
                    
                # Gather the losses across all processes for logging 
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Also gather:
                # - model MSE vs reference MSE (useful to observe divergent behavior)
                # - Implicit accuracy
                if args.train_method != 'sft':
                    avg_model_mse = accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                    avg_ref_mse = accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                    avg_acc = accelerator.gather(implicit_acc).mean().item()
                    implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps
                    avg_win_diff_loss = accelerator.gather((model_losses_w - ref_losses_w).mean().repeat(args.train_batch_size)).mean().item()
                    avg_lose_diff_loss = accelerator.gather((model_losses_l - ref_losses_l).mean().repeat(args.train_batch_size)).mean().item()
                    avg_lose_l_scale = accelerator.gather((lose_l_scale if isinstance(lose_l_scale, torch.Tensor) else loss.new_tensor(lose_l_scale)).mean().repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_adafactor: # Adafactor does itself, maybe could do here to cut down on code
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if args.train_method != 'sft':
                    accelerator.log({"model_mse_unaccumulated": avg_model_mse}, step=global_step)
                    accelerator.log({"ref_mse_unaccumulated": avg_ref_mse}, step=global_step)
                    accelerator.log({"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step)
                    accelerator.log({"avg_win_diff_loss": avg_win_diff_loss}, step=global_step)
                    accelerator.log({"avg_lose_diff_loss": avg_lose_diff_loss}, step=global_step)
                    accelerator.log({"avg_lose_l_scale": avg_lose_l_scale}, step=global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        try:
                            accelerator.save_state(save_path)
                        except FileExistsError as e:
                            logger.info(e)
                        logger.info(f"Saved state to {save_path}")

                        if torch.distributed.is_initialized():
                            torch.distributed.barrier()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.train_method != 'sft':
                logs["implicit_acc"] = avg_acc
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        lora_mixin = StableDiffusionXLLoraLoaderMixin if args.sdxl else StableDiffusionLoraLoaderMixin
        lora_mixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
        )
        logger.info(f"LoRA weights saved to {args.output_dir}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
