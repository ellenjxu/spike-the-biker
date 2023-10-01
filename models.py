import torch
import torch.nn as nn
from torchvision import transforms
import timm

class EfficientNet(nn.Module):
    def __init__(self, out_size: int = 40, out_scale: float = 1, freeze_base: bool = True):
        super().__init__()
        self.out_scale = out_scale
        
        # Load the pre-trained EfficientNet model (for this example, we'll use efficientnet_b0)
        eff_net = timm.create_model('efficientnet_b0', pretrained=True)

        if freeze_base:
            for param in eff_net.parameters():
                param.requires_grad = False

        # replace the last fully connected layer
        num_features = eff_net.classifier.in_features
        eff_net.classifier = nn.Linear(num_features, out_size)
        self.model = nn.Sequential(eff_net, nn.Tanh())

    def forward(self, x):
        x = self.model(x) * self.out_scale  # allows predicting up to out_scale meters away
        # return x.view(-1, 10, 4)  # reshaping the output to [batch_size, 10, 3]
        return x