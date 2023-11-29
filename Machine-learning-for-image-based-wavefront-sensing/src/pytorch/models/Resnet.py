import torch
import numpy as np
import aotools
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):

    def __init__(self, label_type, input_type, model_size, p_drop=0.25, n_zernike=10, npx = 128):
        super(Net, self).__init__()

        if model_size == 18:
            self.resnet = models.resnet18(pretrained=True)  
        elif model_size == 50:
            self.resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError("model not supported, model_size must be 18 or 50")
 
        if label_type != "zernike" and label_type != "phase":
            raise ValueError("label_type must be 'zernike' or 'phase'")

        if input_type != "psf" and input_type != "image":
            raise ValueError("input_type must be 'psf' or 'image'")

        self.label_type = label_type
        self.input_type = input_type

        self.layer_norm = nn.LayerNorm([1, npx, npx])
        
        for param in self.resnet.parameters():
            param.requires_grad = True

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

       
        # Input size 2x128x128 -> 2x224x224
        if self.input_type == "psf":
            first_conv_layer = [nn.Conv2d(1, 3, kernel_size=1, stride=1, bias=True),
                                nn.AdaptiveMaxPool2d(224),
                                self.resnet.conv1]

            self.resnet.conv1= nn.Sequential(*first_conv_layer)
            self.resnet.conv1.apply(init_weights)

        # Fit classifier
        self.resnet.avgpool = nn.Sequential(
                                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                nn.Dropout(p = p_drop),
                            )  
        if model_size == 18:
            self.resnet.fc = nn.Linear(512, n_zernike)
            
                                
        else:
            self.resnet.fc = nn.Linear(2048, n_zernike)

        torch.nn.init.xavier_uniform_(self.resnet.fc.weight)
                               
        
        self.phase2dlayer = Phase2DLayer(n_zernike,npx)


    def forward(self, x):

        x = self.layer_norm(x)
        z = self.resnet(x)

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
