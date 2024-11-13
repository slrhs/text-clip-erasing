import gc
import os
import time
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import trange
from collections import OrderedDict
import random

import utils


class Configuration:
    def __init__(self, *args, **kwargs) -> None:
        args = args[0]
        self.target_concept = args["concept"]
        self.optimizer_name = args["optim"]
        self.lr = args["lr"]
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.eps = args["eps"]
        self.weight_decay = args["weight_decay"]
        self.batch_size = args["batch_size"]
        self.gpu_id = args["gpu_id"]
        self.save_path = args["save"]
        self.num_epoch = args["epochs"]
        self.lambda_reg = args["lambda_reg"]
        self.pos = args["pos"]
        self.neg = args["neg"]
        self.clip_version = args["clip_ver"]
        self.text_encoder_path = args["text_encoder_path"]
        self.tokenizer_version = args["tokenizer_version"]
        self.diffusion_path = args["diffusion_path"]
        


def get_text_embeddings(encoder, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    embeddings = encoder(**inputs).last_hidden_state.mean(dim=1)  
    return embeddings


def load_other_concepts_embeddings(file_paths, tokenizer, encoder, target_concept, device):
    other_concept_list = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            concept = row['concept']
            if str(concept) == target_concept:
                continue
            other_concept_list.append(str(concept))
    return other_concept_list


def esd_loss(student_emb, teacher_emb, empty_emb, device, pos, neg):
    student_emb = student_emb.to(device)
    empty_emb = empty_emb.to(device)
    teacher_emb = teacher_emb.to(device)
    
    
    target_embedding = pos * empty_emb - neg * teacher_emb
    
    loss = F.mse_loss(student_emb, target_embedding)

    return loss

def esd_other_loss(student_emb, teacher_emb, empty_emb, other_teacher_emb, other_student_emb, device, pos, neg, lambda_reg):
    student_emb = student_emb.to(device)
    empty_emb = empty_emb.to(device)
    teacher_emb = teacher_emb.to(device)
    other_teacher_emb = other_teacher_emb.to(device)
    other_student_emb = other_student_emb.to(device)
    
    
    target_embedding = pos * empty_emb - neg * teacher_emb
    base_loss = F.mse_loss(student_emb, target_embedding)
    reg_loss = F.mse_loss(other_teacher_emb, other_student_emb)
    total_loss = base_loss + lambda_reg * reg_loss

    return total_loss

    
      

def train(config: Configuration):
    target_concept = config.target_concept
    device = f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    save_path = config.save_path
    num_epochs = config.num_epoch
    pos = config.pos
    neg = config.neg
    lambda_reg = config.lambda_reg
    
    

    t_tokenizer, teacher_encoder, _, _, _ = utils.load_models_from_local_optioned_path(
        text_encoder_path=config.text_encoder_path,
        unet_path=f"{config.diffusion_path}/unet",
        vae_path=f"{config.diffusion_path}/vae",
        tokenizer_version=config.tokenizer_version,
    )

    s_tokenizer, student_encoder, _, _, _ = utils.load_models_from_local_optioned_path(
        text_encoder_path=config.text_encoder_path,
        unet_path=f"{config.diffusion_path}/unet",
        vae_path=f"{config.diffusion_path}/vae",
        tokenizer_version=config.tokenizer_version,
    )

    other_data_paths = [
        f"/home/david/hyr/erasing/autodl-tmp/text-clip-erasing/text-clip-dataset/concept.csv"
    ]
    other_concept_list = load_other_concepts_embeddings(other_data_paths, t_tokenizer, teacher_encoder, target_concept, device)
    print(other_concept_list)

    for param in teacher_encoder.parameters():
        param.requires_grad = False

    for param in student_encoder.parameters():
        param.requires_grad = True

    optimizer = utils.get_optimizer(
        student_encoder.parameters(),
        config.optimizer_name,
        config.lr,
        (config.beta1, config.beta2),
        config.weight_decay,
        config.eps,
    )

    history = {"loss": []}
    os.makedirs(save_path, exist_ok=True)
    start = time.perf_counter()

    teacher_embedding = get_text_embeddings(teacher_encoder, t_tokenizer, target_concept).to(device)
    empty_embedding = get_text_embeddings(teacher_encoder, t_tokenizer, "").to(device)
    other_concept = random.choice(other_concept_list)
    other_teacher_embedding = get_text_embeddings(teacher_encoder, t_tokenizer, other_concept).to(device)
    
    pbar = trange(0, num_epochs, desc="Epoch")
    for epoch in pbar:
        loss_avg = 0
        cnt = 0
        student_encoder.train()


        student_embedding = get_text_embeddings(student_encoder, s_tokenizer, target_concept).to(device)


        other_student_embedding = get_text_embeddings(student_encoder, s_tokenizer, other_concept).to(device)
        loss = esd_other_loss(student_embedding, teacher_embedding, empty_embedding, other_teacher_embedding, other_student_embedding, device, pos, neg, lambda_reg)

        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        loss_avg += loss.detach().item()
        cnt += 1

        history["loss"].append(loss.detach().item())
        

    pbar.set_postfix(OrderedDict(loss=loss_avg / (cnt + 1e-9)))
    student_encoder.eval()
    concept_no_space = target_concept.replace(" ", "")
    student_encoder.save_pretrained(f"{save_path}/{concept_no_space}-epoch_{epoch}-reg_{lambda_reg}-pos_{pos}-neg_{neg}")
    end = time.perf_counter()
    print(f"Time : {end - start}")

    utils.plot_loss(history, save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, required=True, help="Concept to erase. For example, 'Sunflower'")
    parser.add_argument("--optim", type=str, default="AdamW", choices=["Adam", "AdamW", "Adadelta", "Adagrad"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=31)
    parser.add_argument("--pos", type=int, default=2)
    parser.add_argument("--neg", type=int, default=1)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save", type=str, default="networks")
    parser.add_argument("--clip_ver", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--text_encoder_path", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--tokenizer_version", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--diffusion_path", type=str, default="CompVis/stable-diffusion-v1-4")
    args = vars(parser.parse_args())
    config = Configuration(args)

    train(config=config)
