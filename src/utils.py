# model/utils.py
import torch
from torchvision.models.segmentation import fcn_resnet50

def create_model(num_classes=5, pretrained_backbone=True):
    """
    Returns a torchvision segmentation model (FCN) with the requested output channels.
    num_classes: e.g. 5 = background + 4 defect types
    """
    model = fcn_resnet50(pretrained=False, num_classes=num_classes)
    # If you want pretrained backbone weights, you can load them separately.
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
