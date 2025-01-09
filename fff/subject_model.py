import torch
from fff.utils.truncate import Truncate
from fff.fif import FreeFormInjectiveFlow
from fff.fff import FreeFormFlow
import os
from warnings import warn

class SubjectModel(torch.nn.Module):
    def __init__(self, subject_model_path, model_type=None, truncate=False):
        super(SubjectModel, self).__init__()
    
        if subject_model_path is None:
            self.model = None
            warn("No subject model path given, continuing without subject model")
            return
        
        if not os.path.exists(subject_model_path):
            f"Subject model path {subject_model_path} given, but does not exist"

        if model_type == "FreeFormFlow":
            self.model = FreeFormFlow.load_from_checkpoint(subject_model_path)
            self.model.eval()
        elif model_type == "FreeFormInjectiveFlow":
            self.model = FreeFormInjectiveFlow.load_from_checkpoint(subject_model_path)
            self.model.eval()
        elif model_type == "PrecompiledModel":
            self.model = torch.load(subject_model_path)
            self.model.eval()
        elif model_type == None:
            model_type, self.model = infer_and_load_model_type(subject_model_path)
            self.model.eval()        
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        
        if truncate:
            self.model = Truncate(self.model)

    def forward(self, x, *c, **kwargs):
        if self.model is None:
            raise RuntimeError("No subject model loaded")
        return self.model(x, *c, **kwargs)

    def encode(self, x, *c, **kwargs):
        if self.model is None:
            raise RuntimeError("No subject model loaded")
        return self.model.encode(x, *c, **kwargs)
    
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
            model = torch.load(subject_model_path)
            if hasattr(model, "encode") and hasattr(model, "decode"):
                return "PrecompiledModel", model
            else:
                raise NotImplementedError("Model type not implemented")
        except:
            raise NotImplementedError("Model type not implemented")
