import torch
import torch.nn as nn
from fff.utils.truncate import Truncate
from fff.fif import FreeFormInjectiveFlow
from fff.fff import FreeFormFlow
from fff.some_model import SomeModel
from ldctbench.hub import load_model
import os
from warnings import warn
import torch.nn.functional as F
from fff.model.utils import guess_image_shape
from math import prod


from fff.data.utils import Decolorize

class SubjectModel(torch.nn.Module):
    def __init__(self, subject_model_path, model_type=None, truncate=False, fixed_transform=None):
        super(SubjectModel, self).__init__()

        if model_type in ["cnn10", "redcnn", "wganvgg", "dugan"]:
            self.model = nn.Sequential(
                nn.Unflatten(-1, (1, 128, 128)),
                load_model(model_type, eval=True),
                nn.Flatten(),
            )
            return

        if subject_model_path is None:
            self.model = None
            warn("No subject model path given, continuing without subject model")
            return
        if not os.path.exists(subject_model_path):
            f"Subject model path {subject_model_path} given, but does not exist"

        if model_type == "FreeFormFlow":
            self.model = FreeFormFlow.load_from_checkpoint(subject_model_path)
            self.model.eval()
        elif model_type == "SomeModel":
            self.model = SomeModel.load_from_checkpoint(subject_model_path)
            self.model.eval()
        elif model_type == "FreeFormInjectiveFlow":
            self.model = FreeFormInjectiveFlow.load_from_checkpoint(subject_model_path)
            self.model.eval()
        elif model_type == "PrecompiledModel":
            self.model = torch.load(subject_model_path, weights_only=False)
            self.model.eval()
        elif model_type == None:
            model_type, self.model = infer_and_load_model_type(subject_model_path)
            self.model.eval()
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")

        if truncate:
            self.model = Truncate(self.model)

        for param in self.model.parameters():
            param.requires_grad = False

        if fixed_transform is not None:
            if fixed_transform == "decolorize":
                self.fixed_transform = Decolorize
            else:
                raise NotImplementedError(f"You have to implement {fixed_transform} in subject_model.py")
        else:
            self.fixed_transform = None

    def forward(self, x, *c, **kwargs):
        if self.fixed_transform is not None:
            x = self.fixed_transform(x)
        if self.model is None:
            raise RuntimeError("No subject model loaded")
        # return self.model(x, *c, **kwargs)
        return self.model(x)

    def encode(self, x, *c, **kwargs):
        if self.fixed_transform is not None:
            x = self.fixed_transform(x)
        if self.model is None:
            raise RuntimeError("No subject model loaded")
        try:
            return self.model.encode(x, *c, **kwargs)
        except:
            return self.forward(x, *c, **kwargs)

    
    def decode(self, z, *c, **kwargs):
        if self.model is None:
            raise RuntimeError("No subject model loaded")
        return self.model.decode(z, *c, **kwargs)


def infer_and_load_model_type(subject_model_path):
    if subject_model_path.endswith(".ckpt"):
        try:
            model = FreeFormFlow.load_from_checkpoint(subject_model_path)
            return "FreeFormFlow", model
        except:
            model = FreeFormInjectiveFlow.load_from_checkpoint(subject_model_path)
            return "FreeFormInjectiveFlow", model
    else:
        try:
            model = torch.load(subject_model_path, weights_only=False)
            if hasattr(model, "encode") and hasattr(model, "decode"):
                return "PrecompiledModel", model
            else:
                raise NotImplementedError("Model type not implemented")
        except:
            raise NotImplementedError("Model type not implemented")


class BiomedClipModel(torch.nn.Module):
    def __init__(self, model, tokenizer, image_only=True):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.image_only = image_only

    def encode(self, x, c=None):
        if x.ndim == 2:
            x = x.reshape(x.shape[0], *guess_image_shape(prod(x.shape[1:])))
        if self.image_only:
            text = self.tokenizer(["" for i in range(len(x))], context_length=0).to(
                x.device
            )
        else:
            text = self.tokenizer(c, context_length=256).to(x.device)

        images = self.preprocess(x)
        image_features, text_features, logit_scale = self.model(images, text)
        if self.image_only:
            return image_features
        return image_features, text_features

    def decode(self, x, c=None):
        raise (RuntimeError("BiomedClip has no decoder"))

    def preprocess(self, img):
        _, ch, h, w = img.shape

        if h < w:
            new_h, new_w = 224, int(w * 224 / h)  # Scale width
        else:
            new_w, new_h = 224, int(h * 224 / w)  # Scale height

        img = F.interpolate(
            img,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        # Center crop manually
        top = (new_h - 224) // 2
        left = (new_w - 224) // 2

        img = img[..., top : top + 224, left : left + 224]

        # Convert to RGB (if needed)
        if ch == 1:  # Grayscale input
            img = img.repeat(1, 3, 1, 1)  # Expand to RGB channels

        # Normalize
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=img.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=img.device
        ).view(1, 3, 1, 1)
        img = (img - mean) / std

        return img
