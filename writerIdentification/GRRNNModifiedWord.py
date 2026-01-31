  
import torch
import torch.nn as nn
import math
import timm
import torch.nn as nn

import torchvision.models as models

# 1,num_classes=num_class,mode=self.mode
class VGGnet1(nn.Module):
    def __init__(self, input_channel,num_classes):
        super().__init__()
        layers = [64, 128, 256, 512]
        #self.resnet = models.resnet18(pretrained=True)
        
        self.conv = nn.Conv2d(3,1, kernel_size=3,stride=1, padding=1, bias=False)
        self.resnet = models.resnet18(pretrained = False)#models.resnet18(pretrained=True)
        #self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        pretrained="/cluster/datastore/aniketag/writerClassification/writer-identification/resnet18-5c106cde.pth"
        self.resnet.load_state_dict(torch.load(pretrained))
        print("\n\t pretrained weight loaded:",pretrained)
        
    
        self.resnet.conv1 = self._conv(input_channel, layers[0]) 
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.featv = nn.AdaptiveAvgPool2d((1, 2))
        self.avg = nn.AdaptiveAvgPool2d(1)

        #self.linear0 = nn.Linear(1000,784)
        self.linear = nn.Linear(1000, num_classes)

    def _conv(self, inplanes, outplanes, nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplanes, outplanes, kernel_size=3,
                                  stride=1, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(outplanes))
            conv.append(nn.ReLU(inplace=True))
            inplanes = outplanes

        conv = nn.Sequential(*conv)

        return conv

    def forward(self, x):
        #print("\n\t 1.x.shape:",x.shape)
        
        x = self.conv(x)
        x = self.resnet(x)
        
        intermediate_output = x.clone()
        #x = self.linear0(x)
        x = self.linear(x)
        
        return x,intermediate_output



import torch.nn.functional as F

from safetensors.torch import safe_open
import torch.nn as nn
import timm
from torchvision import models

class mobilenet_v2(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', num_classes=0, pretrained=True, trainable=True
    ):
        super().__init__()
        
        #self.model = models.resnet50(pretrained=False)
        self.model = models.mobilenet_v2(pretrained=False)

        #weights_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/model/model.safetensors"
        #weights_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/model/resnet50-0676ba61.pth"

        weights_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/model/mobilenet_v2-b0353104.pth"
        # Load weights from the safetensors file if provided
        if weights_path:
                
            state_dict = torch.load(weights_path, map_location='cuda:1' if torch.cuda.is_available() else 'cpu')
            #self.resnet50.load_state_dict(state_dict)
       
            self.model.load_state_dict(state_dict)
            print("\n\t pretrained weights loaded successfully from ",weights_path)        
    
        for p in self.model.parameters():
            p.requires_grad = trainable
            
        self.final_layer = nn.Linear(1000, num_classes)
         
    def forward(self, x):
        
        #print("\n\t 1.x.shape:",x.shape)
        
        if 0:#x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Repeat the 1 channel to create 3 channels
        
        #print("\n\t 2.x.shape:",x.shape)
        x = self.model(x)
        intermediate_output = x.clone()
        x = self.final_layer(x)
        return x,intermediate_output  

class VGGnet3(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', num_classes=0, pretrained=True, trainable=True
    ):
        super().__init__()
        """
        self.model = timm.create_model(
            model_name, pretrained, num_classes=num_classes, global_pool="max"
        )
        """
        self.model = timm.create_model(
            model_name, pretrained=False, num_classes=1000, global_pool="max"
        )
        
        #self.model = models.resnet50(pretrained=False)
        self.model = models.mobilenet_v2(pretrained=False)

        #weights_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/model/model.safetensors"
        #weights_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/model/resnet50-0676ba61.pth"

        weights_path = "/cluster/datastore/aniketag/writerClassification/writer-identification/model/mobilenet_v2-b0353104.pth"
        # Load weights from the safetensors file if provided
        if weights_path:
            try:
                with safe_open(weights_path, framework="pt", device="cuda:1") as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
            except Exception as e:
                print("exceptin!!!")
                
            state_dict = torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            #self.resnet50.load_state_dict(state_dict)
       
            self.model.load_state_dict(state_dict)
            print("\n\t pretrained weights loaded successfully!!!")        
    
        #self.model = torch.compile(self.model, backend="inductor")
        for p in self.model.parameters():
            p.requires_grad = trainable
            
        self.final_layer = nn.Linear(1000, num_classes)
         
    def forward(self, x):
        
        #print("\n\t 1.x.shape:",x.shape)
        
        if 0:#x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Repeat the 1 channel to create 3 channels
        
        #print("\n\t 2.x.shape:",x.shape)
        x = self.model(x)
        intermediate_output = x.clone()
        x = self.final_layer(x)
        return x,intermediate_output  


class VGGnet2(nn.Module):
    def __init__(self, input_channel,num_classes):
        super().__init__()
        layers = [64, 128, 256, 512]
        #self.resnet = models.resnet18(pretrained=True)
        self.resnet = models.resnet34(pretrained = True)#models.resnet18(pretrained=True)
        #self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        pretrained="/cluster/datastore/aniketag/writerClassification/writer-identification/resnet18-5c106cde.pth"
        self.resnet.load_state_dict(torch.load(pretrained))
        
        self.resnet.conv1 = self._conv(input_channel, layers[0]) 
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.featv = nn.AdaptiveAvgPool2d((1, 2))
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.linear0 = nn.Linear(1000,784)
        self.linear = nn.Linear(784, num_classes)

    def _conv(self, inplanes, outplanes, nlayers=3):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplanes, outplanes, kernel_size=3,
                                  stride=1, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(outplanes))
            conv.append(nn.ReLU(inplace=True))
            inplanes = outplanes

        conv = nn.Sequential(*conv)

        return conv

    def forward(self, x):
        print("\n\t 1.x.shape:",x.shape)
        x = self.resnet(x)
        x = self.linear0(x)
        intermediate_output = x.clone()
        x = self.linear(x)
        return x,intermediate_output




if __name__ == '__main__':
    
    x = torch.rand(256,1,64,256)
    
    #mod = GrnnNet(1,105,mode='vertical')
    
    #logits = mod(x)
    
    #print(logits.shape)
    
    #model = VGGnet2(1,672)

    if 0:
        model = VGGnet2(1,672)
        dummy_input = torch.randn(3, 1, 64, 256)

        # Forward pass
        feat,intermediate_output = model(dummy_input)

        # Print the output shapes
        #print("glf shape:", glf.shape)
        print("feat shape:", feat.shape)
        print("\n\t intermediate_output.shape:",intermediate_output.shape,"\t dummy_input.shape:",dummy_input.shape)
    


    dummy_input = torch.randn(1, 3, 224, 224)  # For a single image with size 224x224 and 3 color channels
    
    # Instantiate the VGGnet3 model
    model = VGGnet3(model_name='resnet50', num_classes = 672,pretrained=True, trainable=True)
    
    # Forward pass with dummy input
    output = model(dummy_input)
    
    # Print the output shape
    print("1. Output shape:", output.shape)
