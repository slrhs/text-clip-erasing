import os
import random
import argparse
import warnings
import pandas as pd

import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, PNDMScheduler


warnings.filterwarnings('ignore')

def generate_images(prompt, seed, case_number, num_images_per_prompt, pipe, save_dir):
    device = "cuda:0"
    generator = torch.Generator(device).manual_seed(seed)
    images = pipe(
        prompt,
        num_inference_steps=100,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator
    ).images
    
    model_path_clean = args.text_encoder_path.replace('networks/', '')
    parts = model_path_clean.split('-')
    model_name_clean = f"{parts[0]}-{parts[1]}-{parts[2]}-{parts[3]}-{parts[4]}"
    
    folder_path = os.path.join(save_dir, model_name_clean)
    os.makedirs(folder_path, exist_ok=True)

    for i in range(len(images)):
        # Use case_number and sequence number for naming
        image_filename = f"{case_number}_{i}.png"
        images[i].save(os.path.join(folder_path, image_filename))
    
    print(f"{prompt}")

def main(args):
    device = "cuda:0"

    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_path).to(device)

    scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1
    )

    if args.model_name is None:
        vae = AutoencoderKL.from_pretrained(args.vae_path)
        unet = UNet2DConditionModel.from_pretrained(args.unet_path)
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)

        pipe = StableDiffusionPipeline(
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.model_name).to(device)
        pipe.text_encoder = text_encoder

    # Read the dataset and generate images
    df = pd.read_csv(args.dataset_path)
    for _, row in df.iterrows():
        prompt = row['prompt']
        seed = row['evaluation_seed']
        case_number = row['case_number']
        num_images_per_prompt = args.num_images_per_prompt  # Set this as needed
        generate_images(prompt, seed, case_number, num_images_per_prompt, pipe, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CSV dataset')
    parser.add_argument('--text_encoder_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="/home/david/hyr/erasing/autodl-tmp/stable-diffusion-main/stable-diffusion-v1-4")
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--unet_path', type=str)
    parser.add_argument('--vae_path', type=str)
    parser.add_argument('--num_images_per_prompt', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default="exp")
    args = parser.parse_args()
    
    main(args)
