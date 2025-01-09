import torch
from fff.utils.truncate import Truncate
from fff.fif import FreeFormInjectiveFlow
from fff.fff import FreeFormFlow


class SubjectModel(torch.nn.Module):
    def __init__(self, subject_model_path, model_type='FreeFormFlow', truncate=False):
        super(SubjectModel, self).__init__()
        if model_type == "FreeFormFlow":
            self.model = FreeFormFlow.load_from_checkpoint(subject_model_path)
            self.model.eval()
        elif model_type == "FreeFormInjectiveFlow":
            self.model = FreeFormInjectiveFlow.load_from_checkpoint(subject_model_path)
            self.model.eval()
        else:
            raise NotImplementedError("Model type not implemented")
        
        if truncate:
            self.model = Truncate(self.model)

    def forward(self, x, *c, **kwargs):
        return self.model(x, *c, **kwargs)

    def encode(self, x, *c, **kwargs):
        return self.model.encode(x, *c, **kwargs)
    
    def decode(self, z, *c, **kwargs):
        return self.model.decode(z, *c, **kwargs)