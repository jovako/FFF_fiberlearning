#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fff.subject_model import SubjectModel
from transformers import AutoModelForImageClassification
from fff.data import load_dataset
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from safetensors.torch import load_file
import sys
sys.path.append("/home/hd/hd_hd/hd_gu452/FFF_fiberlearning/scripts/")
from fff.ndtm import NDTMConfig, NDTMTimestepCompatability, DiffusionScheduleConfig, StableDiffusionInterface, DiffusionSchedule, DiffusionModel, NDTM
sys.path.append("/home/hd/hd_hd/hd_gu452/oc-guidance/")
from utils.functions import get_timesteps
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
from datetime import datetime
import random

# Constants
global_mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
global_std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
image_size = 384
to_grayscale = True
batch_size = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ClassifierOutput(ModelOutput):
    loss: torch.Tensor = None
    logits: torch.Tensor = None

#class BiomedClipClassifier(nn.Module):
#    def __init__(self, pretrained_path="../SubjectModels/saved_models/BiomedClip.pt", num_labels=5, dropout=0.1):
#        super().__init__()
#
#        # 1️⃣ Load pretrained CLIP
#        self.biomedclip = torch.load(pretrained_path, weights_only=False)
#        self.biomedclip.fixed_transform = None
#        self.biomedclip.empty_condition = True
#        # Freeze CLIP parameters
#        for p in self.biomedclip.parameters():
#            p.requires_grad = False
#
#        # 2️⃣ Define classification head
#        embed_dim = self.biomedclip.model.visual.head.proj.out_features
#        self.classifier = nn.Sequential(
#            nn.LayerNorm(embed_dim),
#            nn.Linear(embed_dim, embed_dim // 2),
#            nn.ReLU(),
#            nn.Dropout(dropout),
#            nn.Linear(embed_dim // 2, num_labels)
#        )
#
#    def forward(self, pixel_values=None, labels=None, **kwargs):
#        features = self.biomedclip.encode(pixel_values)
#
#        # 4️⃣ Forward through classification head
#        logits = self.classifier(features)
#
#        # 5️⃣ Optionally compute loss
#        loss = None
#        if labels is not None:
#            loss = nn.CrossEntropyLoss()(logits, labels)
#        return ClassifierOutput(loss=loss, logits=logits)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)
    

class ResNetClassifier(nn.Module):
    def __init__(self, embed_dim, num_labels, blocks=2, dropout=0.0, hidden_dim_factor=4):
        super().__init__()
        hidden_dim = int(embed_dim * hidden_dim_factor)

        # One or more residual blocks; add more if desired
        self.blocks = nn.ModuleList()
        for _in in range(blocks):
            self.blocks.append(ResidualMLPBlock(embed_dim, hidden_dim, dropout))

        # Final classification head
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_labels),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x)


class BiomedClipClassifier(nn.Module):
    def __init__(self, pretrained_path="../SubjectModels/saved_models/BiomedClip.pt", num_labels=5, dropout=0.05):
        super().__init__()

        # 1️⃣ Load pretrained CLIP
        self.biomedclip = torch.load(pretrained_path, weights_only=False)
        self.biomedclip.fixed_transform = None
        self.biomedclip.empty_condition = True
        # Freeze CLIP parameters
        for p in self.biomedclip.parameters():
            p.requires_grad = False

        # 2️⃣ Define classification head
        embed_dim = self.biomedclip.model.visual.head.proj.out_features
        self.classifier = ResNetClassifier(embed_dim, num_labels, blocks=2, dropout=dropout)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        with torch.no_grad():
            features = self.biomedclip.encode(pixel_values)

        # 4️⃣ Forward through classification head
        logits = self.classifier(features)

        # 5️⃣ Optionally compute loss
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return ClassifierOutput(loss=loss, logits=logits)


class BiomedClipSubjectModel(nn.Module):
    def __init__(self, model_path=f'biomedclip-pretrained-larger-chexpert_{image_size}_2'):
        super().__init__()
        self.model = BiomedClipClassifier()
        weights = load_file(os.path.join(model_path, "model.safetensors"))
        self.model.load_state_dict(weights)
        self.model.eval()
        
    def forward(self, x):
        return self.model(x.repeat(1, int(3/n_channels), 1, 1)).logits

    def decode(self, y):
        raise NotImplementedError("DINOv2 does not support decoding.")


class ConvNextClassfierSubjectModel(nn.Module):
    def __init__(self, model_path=f'convnextv2-tiny-chexpert_{image_size}_2'):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        
    def forward(self, x):
        return self.model(x.repeat(1, int(3/n_channels), 1, 1)).logits

    def decode(self, y):
        raise NotImplementedError("DINOv2 does not support decoding.")

def normalize(img, value_range=[0, 1]):
    #Bring to 0, 1
    img = (img + value_range[0])/(value_range[1] - value_range[0])
    img = (img - global_mean.to(img.device)) / global_std.to(img.device)
    return img

def denormalize(img, clamp=True, value_range=[0, 1]):
    img = img * global_std.to(img.device) + global_mean.to(img.device)
    # Bring into value_range
    img = img*(value_range[1] - value_range[0]) + value_range[0]
    if clamp:
        img = torch.clamp(img, *value_range)
    return img

if __name__ == "__main__":
    #Derived parameters
    n_channels = 1 if to_grayscale else 3
    if to_grayscale:
        global_mean, global_std = global_mean.mean(dim=1, keepdims=True), global_std.mean(dim=1, keepdims=True)
    normalized_boundaries = normalize(torch.tensor([0, 1]).reshape(1, 1, 2).repeat(1, n_channels, 1)).cpu().detach().numpy()
    value_range = (normalized_boundaries.min(), normalized_boundaries.max())

    # Configs
    timestep_config = NDTMTimestepCompatability()
    diffusion_schedule_config = DiffusionScheduleConfig()
    NDTM_config = NDTMConfig(N=4, 
                             gamma_t= lambda t: 10 if any(t < 400) else 0.2 / (t[0].item()/600), # torch.sigmoid((800 - t)/400) * 20.0, 
                             u_lr=0.002, 
                             w_terminal=3.0, 
                             eta=0.5,
                             u_lr_scheduler="linear",
                             w_score_scheme="zero",
                             w_control_scheme="ones",
                             clip_images=True,
                             clip_range=value_range,
                             ancestral_sampling=False,
                             variance_type="large")
    data_set_config = {
        "name": "chexpert",
        "root": "/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-chexpert",
        "patchsize": None,
        "resize_to": image_size,
        "to_grayscale": to_grayscale,
    }
    
    
    # Diffusion Model
    generative_model_chkpt_path = "diffusion-chexpert-cifar-scheduler/epoch_5"
    base_model = StableDiffusionInterface(generative_model_chkpt_path)
    diffusion_schedule = DiffusionSchedule(diffusion_schedule_config)
    generative_model = DiffusionModel(base_model, diffusion_schedule, class_cond_diffusion_model=False)
    
    # Subject Models
    subject_model_convnext = ConvNextClassfierSubjectModel().to(device)
    subject_model_biomed = BiomedClipSubjectModel().to(device)

    # Dataset
    _, val_ds, _ = load_dataset(**data_set_config)
    dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    #Guidance
    ndtm_convnext = NDTM(
        generative_model=generative_model,
        subject_model=subject_model_convnext,
        hparams=NDTM_config
    )
    ndtm_biomed = NDTM(
        generative_model=generative_model,
        subject_model=subject_model_biomed,
        hparams=NDTM_config
    )

    # Start to sample invariances
    invariances_convnext = []
    invariances_biomed = []
    originals = []
    labels = []
    
    invariances_convnext_embeddings = []
    invariances_biomed_embeddings = []
    original_convnext_embeddings = []
    original_biomed_embeddings = []
    
    invariances_convnext_cross_embeddings = []
    invariances_biomed_cross_embeddings = []

    start_time = datetime.now().strftime('%H_%M_%S__%d_%m_%Y')
    random_suffix = start_time + "_" + str(random.getrandbits(16))
    filename = f"sampled_invariances_v2_{random_suffix}.pt"
    
    if os.path.exists(filename):
        raise(RuntimeError("Incredible..."))
    
    for n_batch, batch in enumerate(dataloader):
        x = batch[0].to(device)
        labels.append(batch[1])
        # if n_batch >=1:
        #     break
        with torch.no_grad():
            test_image_embedding_convnext = subject_model_convnext(x)
            test_image_embedding_biomed = subject_model_biomed(x)
        originals.append(x)
        original_convnext_embeddings.append(test_image_embedding_convnext)
        original_biomed_embeddings.append(test_image_embedding_biomed)
        
        ts = get_timesteps(NDTMTimestepCompatability())
        imgs_noised, imgs_approximated = ndtm_convnext.sample(x, None, ts, y_0 = test_image_embedding_convnext.to(device))
        invariances_convnext.append(imgs_noised[0])
        with torch.no_grad():
            invariances_convnext_embeddings.append(subject_model_convnext(imgs_noised[0].to(device)))
            invariances_convnext_cross_embeddings.append(subject_model_biomed(imgs_noised[0].to(device)))    
    
        
        ts = get_timesteps(NDTMTimestepCompatability())
        imgs_noised, imgs_approximated = ndtm_biomed.sample(x, None, ts, y_0 = test_image_embedding_biomed.to(device))
        invariances_biomed.append(imgs_noised[0])
        with torch.no_grad():
            invariances_biomed_embeddings.append(subject_model_biomed(imgs_noised[0].to(device)))
            invariances_biomed_cross_embeddings.append(subject_model_convnext(imgs_noised[0].to(device)))

        torch.save({
            "invariances_convnext": torch.cat(invariances_convnext, dim=0),
            "invariances_biomed": torch.cat(invariances_biomed, dim=0),
            "originals": torch.cat(originals, dim=0),
            "labels": torch.cat(labels, dim=0),
            "invariances_convnext_embeddings": torch.cat(invariances_convnext_embeddings, dim=0),
            "invariances_biomed_embeddings": torch.cat(invariances_biomed_embeddings, dim=0),
            "original_convnext_embeddings": torch.cat(original_convnext_embeddings, dim=0),
            "original_biomed_embeddings": torch.cat(original_biomed_embeddings, dim=0),
            "invariances_convnext_cross_embeddings": torch.cat(invariances_convnext_cross_embeddings, dim=0),
            "invariances_biomed_cross_embeddings": torch.cat(invariances_biomed_cross_embeddings, dim=0),
        }, filename)


# In[ ]:




