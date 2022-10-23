import os
import torch
import numpy as np
from rangin.models import *
from rangin.utils.inference import save_outputs
from utils.inference import *

def infer(input_data, model, pretrained_weights, output_directory):

    device = torch.device(device) if device == "cpu" else torch.device(f"cuda:{device}")
    
    if isinstance(input_data) == str:
        input_data = image_from_path(input_data)
    elif isinstance(input_data) == np.ndarray:
        input_data = torch.tensor(input_data)
    elif isinstance(input_data) == torch.Tensor:
        pass
    else:
        raise NotImplementedError
    
    if input_data.ndim == 3:
        input_data = input_data.unsqueeze(0)
    if input_data.shape[3] == 1:
        input_data = input_data.permute(0, 3, 1, 2)
    
    if not os.path.exists(pretrained_weights):
        print("Pretrained weights file not found")
        raise FileNotFoundError
    chkp = torch.load(pretrained_weights)["model"]
    model.load_state_dict(chkp)

    op = model(input_data)
    save_outputs(op, output_directory)
