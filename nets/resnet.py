import itertools
import os
import sys

sys.path.append("../")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from collections import OrderedDict
from torch.nn import Sequential, Linear, ReLU
from torchvision.models.resnet import ResNet as ResNetTv
from torchvision.models.resnet import Bottleneck, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

class ResNet(ResNetTv):
    """
    A ResNet inheriting from Torchvision's model while allowing for customization of the network head layers.
    Main ResNet body can be either ResNet50, ResNet101 or ResNet152.
    """

    def __init__(self, arch='resnet50', head=[2048, 1000], pretrained=None,
                 first_trainable=0, pretraining_model=None):
        # Initialize main network stages
        if arch=='resnet50':
            super(ResNet, self).__init__(Bottleneck, [3,4,6,3]) 
            in_weights = ResNet50_Weights.IMAGENET1K_V2
        elif arch=='resnet101':
            super(ResNet, self).__init__(Bottleneck, [3,4,23,3]) 
            in_weights = ResNet101_Weights.IMAGENET1K_V2
        elif arch=='resnet152':
            super(ResNet, self).__init__(Bottleneck, [3,8,36,3]) 
            in_weights = ResNet152_Weights.IMAGENET1K_V2
        if pretrained is not None:
            match pretrained.lower():
                case "imagenet":
                    self.load_state_dict(in_weights.get_state_dict(progress=True))
                case "rsp" | "millionaid":
                    # Check pretraining is requested for ResNet50
                    if arch != 'resnet50':
                        raise ValueError("RSP pretraining not supported for networks different from ResNet50.")
                    # Replace head to align to MillionAid classes
                    self.fc = Linear(2048, 51) 
                    # Load state dictionary from checkpoint
                    rsp_state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'weights', 'rsp-resnet-50-ckpt.pth'), weights_only=False)['model']
                    # Actually load state dictionary
                    self.load_state_dict(rsp_state_dict, strict=True) 
                case _:
                    raise ValueError("Cannot pretrain the network. Unknown pretraining weights.")
        # Change classification head
        linears = [(f'linear{i+1}',Linear(head[i], head[i+1], bias=True)) for i in range(len(head)-1)] # Linears in classification head
        relus = [(f'relu{i+1}',ReLU()) for i in range(len(head)-2)] # ReLUs in classification head
        fc_layers = [l for l in list(itertools.chain.from_iterable(itertools.zip_longest(linears,relus))) if l is not None]
        self.fc = Sequential(OrderedDict(fc_layers))
        self.sigmoid = nn.Sigmoid()
        # Save network parameters
        self.num_classes = head[-1]
        if  0 <= first_trainable <= 5:       # sanity check on first trainable 
            self.first_trainable = first_trainable
        else:
            raise ValueError(f'Illegal value ({first_trainable}) for parameter first_trainable. Must be between 0 and 5 (included).')
        # Create initial stage for convenience
        self.layer0 = Sequential(self.conv1,
                                 self.bn1,
                                 self.relu,
                                 self.maxpool)
        # Collect stages in an array (to allow training only some layers)
        self.trainable_layers = [self.layer0,   # 0
                                 self.layer1,   # 1
                                 self.layer2,   # 2
                                 self.layer3,   # 3
                                 self.layer4,   # 4
                                 self.fc]       # 5
        
        # Freeze non-trainable layers by setting parameter requires_grad to false
        for layer in self.trainable_layers[:self.first_trainable]:
            for param in layer.parameters():
                param.requires_grad = False

        # Load weights if pretraining is required
        if pretraining_model is not None:
            self.load_state_dict(torch.load(pretraining_model, weights_only=True), strict=True) 
        
    def forward(self, x):
        
        # Main ResNet section
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Pooling layer
        x = self.avgpool(x)
        # Classification head
        x = x.view((x.shape[0],-1))
        x = self.fc(x)
        pred = self.sigmoid(x)

        return pred