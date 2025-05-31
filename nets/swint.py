import itertools
import os
import sys

sys.path.append("../")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from collections import OrderedDict
from torch.nn import Sequential, Linear, ReLU
from torchvision.models.swin_transformer import SwinTransformer, Swin_T_Weights

class SwinT(SwinTransformer):

    def __init__(self, head=[768, 1000], pretrained=None,
                 first_trainable=0, pretraining_model=None, **kwargs):
        # Initialize Swin-T, parameters from source code to create Swin-T
        super(SwinT, self).__init__(patch_size=[4, 4],
                                        embed_dim=96,
                                        depths=[2, 2, 6, 2],
                                        num_heads=[3, 6, 12, 24],
                                        window_size=[7, 7],
                                        stochastic_depth_prob=0.2,
                                        **kwargs)
        in_weights = Swin_T_Weights.IMAGENET1K_V1
        if pretrained is not None:
            match pretrained.lower():
                case "imagenet":
                    self.load_state_dict(in_weights.get_state_dict(progress=True))
                case "rsp" | "millionaid":
                    # Replace head to align to MillionAid classes
                    self.head = Linear(768, 51) 
                    # Load state dictionary from checkpoint
                    rsp_state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'weights', 'rsp-swin-t-ckpt.pth'), weights_only=False)['model']
                    # Delete 'attn_mask' entries, not present in torchvision implementation
                    attn_masks = [layer for layer in rsp_state_dict if 'attn_mask' in layer]
                    for am in attn_masks:
                        del rsp_state_dict[am]
                    # Flatten 'relative_position_index' entries, torchvision expects size (2401) instead of (49,49)
                    for layer in rsp_state_dict:
                        if 'relative_position_index' in layer:
                            rsp_state_dict[layer] = torch.flatten(rsp_state_dict[layer])
                    # Clone dictionary to allow loading; loading from base dictionary would not work
                    temp_dict = OrderedDict()
                    model_layers = [layer for layer in self.state_dict()]   # List of layers in current model
                    for i, layer in enumerate(rsp_state_dict):
                        assert self.state_dict()[model_layers[i]].shape == rsp_state_dict[layer].shape # Check layer weight size in the 2 dicitonaries
                        temp_dict[model_layers[i]] = rsp_state_dict[layer]
                    # Actually load state dictionary
                    self.load_state_dict(temp_dict, strict=True) 
                case _:
                    raise ValueError("Cannot pretrain the network. Unknown pretraining weights.")
        # Change classification head
        linears = [(f'linear{i+1}',Linear(head[i], head[i+1], bias=True)) for i in range(len(head)-1)] # Linears in classification head
        relus = [(f'relu{i+1}',ReLU()) for i in range(len(head)-2)] # ReLUs in classification head
        fc_layers = [l for l in list(itertools.chain.from_iterable(itertools.zip_longest(linears,relus))) if l is not None]
        self.head = Sequential(OrderedDict(fc_layers))
        self.sigmoid = nn.Sigmoid()
        # Save network parameters
        self.num_classes = head[-1]
        if  0 <= first_trainable <= 5:       # sanity check on first trainable 
            self.first_trainable = max(1, first_trainable)-1 # Custom numbering on trainable layers (see below)
        else:
            raise ValueError(f'Illegal value ({first_trainable}) for parameter first_trainable. Must be between 0 and 5 (included).')
        # Collect stages in an array (to allow training only some layers)
        self.trainable_layers = [self.features[:2],                             # 0 and 1 [First stage]
                                 self.features[2:4],                            # 2 [Second stage]
                                 self.features[4:6],                            # 3 [Third stage]
                                 self.features[6:] + Sequential(self.norm),     # 4 [Fourth stage and BatchNorm]
                                 self.head]                                     # 5 [Classification head]
        
        # Freeze non-trainable layers by setting parameter requires_grad to false
        for layer in self.trainable_layers[:self.first_trainable]:
            for param in layer.parameters():
                param.requires_grad = False

        # Load weights if pretraining is required
        if pretraining_model is not None:
            self.load_state_dict(torch.load(pretraining_model, weights_only=True), strict=True) 
        
    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        pred = self.sigmoid(x)
        return pred       