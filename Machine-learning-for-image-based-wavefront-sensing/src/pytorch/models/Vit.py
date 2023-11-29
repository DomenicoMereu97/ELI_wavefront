import torch
import numpy as np
import aotools
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pyoptica as po
from Cream.EfficientViT.classification.model.build import EfficientViT_M5, EfficientViT_M4

class VitNet(nn.Module):

    def __init__(self, input_type, model_size, p_drop=0.25, n_zernike=10, npx = 128):
        super(Net, self).__init__()

        if model_size == "V4":
            self.vit = EfficientViT_M4(pretrained='efficientvit_m4')
        elif model_size == "V5":
            self.vit = EfficientViT_M5(pretrained='efficientvit_m5')
        else:
            raise ValueError("model not supported, model_size must be V4 or V5")

        self.label_type = label_type
        self.input_type = input_type

        self.layer_norm = nn.LayerNorm([1, npx, npx])
        

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

       
        # Input size 2x128x128 -> 2x224x224
        if self.input_type == "psf":
            first_conv_layer = [nn.Conv2d(1, 3, kernel_size=1, stride=1, bias=True),
                                nn.AdaptiveMaxPool2d(224),
                                self.vit.patch_embed]


            #self.vit.conv1.apply(init_weights)
        else:
            first_conv_layer = [nn.AdaptiveMaxPool2d(224),
                                self.vit.patch_embed]

        self.vit.patch_embed= nn.Sequential(*first_conv_layer)
        #self.vit.conv1.apply(init_weights)

        # Fit classifier
        self.vit.avgpool = nn.Sequential(
                                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                nn.Dropout(p = p_drop),
                            )  
     

        self.vit.head.l = nn.Linear(384, n_zernike)  
                                

        torch.nn.init.xavier_uniform_(self.vit.head.l.weight)
                               
        
        self.phase2dlayer = Phase2DLayer(n_zernike,npx)

        for param in self.vit.parameters():
            param.requires_grad = True


    def forward(self, x):

        x = self.layer_norm(x)
        z = self.vit(x)

        """ if self.label_type == "zern":
            return z """

        phase = self.phase2dlayer(z)
        
        return {'phase': phase, "zernike" : z}

class Phase2D(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, z_basis):
        ctx.z_basis = z_basis.cuda()
        output = input[:,:, None, None] * ctx.z_basis[None, 1:,:,:]
        return torch.sum(output, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        dL_dy = grad_output.unsqueeze(1)
        dy_dz = ctx.z_basis[1:,:,:].unsqueeze(0)
        grad_input = torch.sum(dL_dy * dy_dz, dim=(2,3))
        return grad_input, None
    
class Phase2DLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(Phase2DLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.z_basis = aotools.zernikeArray(input_features+1, output_features, norm='noll')
        self.z_basis = torch.as_tensor(self.z_basis, dtype=torch.float32)
        
    def forward(self, input):
        return Phase2D.apply(input, self.z_basis)     
