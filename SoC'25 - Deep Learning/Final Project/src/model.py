import torch.nn as nn
from torchvision.models import mobilenet_v2

def build_model():
  model = mobilenet_v2(pretrained=True)
  model.classifier[1] = nn.Linear(model.last_channel, 200)
  return model