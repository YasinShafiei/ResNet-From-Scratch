'''
Implementation of the Residual Network.
The resnet paper by Kaiming He et al: 
https://arxiv.org/abs/1512.03385
'''
# importings 
import torch.nn as nn
import torch.nn.functional as F 


### ------------------------------ < RESNET 50, 101, 152 > ------------------------------ ### 
# According to the ResNet paper, the ResNet 50, 101 and 151 has 
# 3 convolutional layers in their residual blocks
class ExpansionResBlock(nn.Module):
    '''
    Residual Block contains 3 conv layers followed by a batch norm
    '''
    def __init__(self, in_channels, inter_channels, downsampling_layer=None, stride=1):
        super().__init__()
        expansion = 4 # in the resnet paper, the number of output features of the last layer 
                      # is 4 times greater than the prevoius layers. i.e 64 will be 256. 

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        
        self.conv2 = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        
        self.conv3 = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels * expansion, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_channels * expansion)
        self.relu = nn.ReLU()
        self.downsampling_layer = downsampling_layer 
    
    def forward(self, x):
        '''
        forward pass 
        '''
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # downsample
        if self.downsampling_layer is not None:
            res = self.downsampling_layer(res)

        out += res 
        out = F.relu(out)

        return out 
    
class LargeResNet(nn.Module):
    '''
    Residual Network 
    '''
    def __init__(self, num_blocks, img_channels, num_classes, use_final=True):
        super().__init__()
        self.expansion = 4
        self.use_final = use_final
        self.in_channels = 64 # starting input size fo residual blocks
                              # This value will increase by the factor of 4 
        
        ### INPUT BLOCK:
        # the input block contains a convolutional block with the output size of 64. 
        # the kernel size for this conv layer is set to 7 (7x7 kernel) with a stride of one 
        # which mean no downsampling 
        self.conv1 = nn.Conv2d(img_channels, 
                               self.in_channels, 
                               kernel_size=7,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        ### RESIDUAL LAYERS:
        # In the resnet paper, there are 4 residual layers,
        # but for each layer, the number of ResBlocks is different (based on the model variant needed)
        self.layer1 = self._layer(ExpansionResBlock, num_blocks[0], stride=1, inter_channels=64)
        self.layer2 = self._layer(ExpansionResBlock, num_blocks[1], stride=2, inter_channels=128)
        self.layer3 = self._layer(ExpansionResBlock, num_blocks[2], stride=2, inter_channels=256)
        self.layer4 = self._layer(ExpansionResBlock, num_blocks[3], stride=2, inter_channels=512)

        ### FINAL LAYERS:
        # if the user decided to use the final layer, it will be used based on the paper has a Linear layer with 
        # 2048 units
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if use_final:
            # the final layers
            self.fc = nn.Linear(512 * self.expansion, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        forward pass
        '''
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        
        if self.use_final:
            out = self.fc(out)

        return out

    def _layer(self, block, num_blocks, stride, inter_channels):
        '''
        making a residual layer containing num_layers number of residual blocks. 
        '''
        layers = []
        downsampling_layer = None # the downsampling will be filled if it matched the condition of:
            # the stride being greater than one or twe want to downsample for identity mapping
        if stride != 1 or self.in_channels != inter_channels * self.expansion:
            downsampling_layer = nn.Sequential(
                nn.Conv2d(self.in_channels, inter_channels * self.expansion, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(inter_channels * self.expansion)
            )
        
        # add the block for downsampling 
        layers.append(block(self.in_channels, inter_channels, downsampling_layer, stride))
        self.in_channels = inter_channels * self.expansion # updating for passing it to the next layer of the block

        # adding blocks
        for _ in range(num_blocks - 1):
            layers.append(block(self.in_channels, inter_channels))

        return nn.Sequential(*layers)

### ------------------------------ < RESNET 18, 34 > ------------------------------ ### 
# According to the ResNet paper, the ResNet 18 and 34 has 
# 2 convolutional layers in their residual blocks
class BasicBlock(nn.Module):
    '''
    Residual Block contains 2 conv layers followed by a batch norm
    '''
    def __init__(self, in_channels, out_channels, downsampling_layer=None, stride=1):
        super().__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU() 
        self.downsampling_layer = downsampling_layer 

    def forward(self, x):
        '''
        forward pass 
        '''
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # downsample
        if self.downsampling_layer is not None:
            res = self.downsampling_layer(res)

        out += res 
        out = self.relu(out)

        return out 
    
class SmallResNet(nn.Module):
    '''
    Residual Network 
    '''
    def __init__(self, num_layers, img_channels, num_classes, use_final=True):
        super().__init__()
        self.expansion = 1
        self.use_final = use_final
        self.in_channels = 64 # starting input size fo residual blocks
                              # This value will increase by the factor of 1
        
        ### INPUT BLOCK:
        # the input block contains a convolutional block with the output size of 64. 
        # the kernel size for this conv layer is set to 7 (7x7 kernel) with a stride of one 
        # which mean no downsampling 
        self.conv1 = nn.Conv2d(img_channels, 
                               64, 
                               kernel_size=7,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU()
        ### RESIDUAL LAYERS:
        # In the resnet paper, there are 4 residual layers,
        # but for each layer, the number of ResBlocks is different (based on the model variant needed)
        self.layer1 = self._layer(BasicBlock, num_layers[0], stride=1, out_channels=64)
        self.layer2 = self._layer(BasicBlock, num_layers[1], stride=2, out_channels=128)
        self.layer3 = self._layer(BasicBlock, num_layers[2], stride=2, out_channels=256)
        self.layer4 = self._layer(BasicBlock, num_layers[3], stride=2, out_channels=512)

        ### FINAL LAYERS:
        # if the user decided to use the final layer, it will be used based on the paper has a Linear layer with 
        # 512 * expansion-rate units
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if use_final:
            # the final layers
            self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        '''
        forward pass
        '''
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.relu(out)
        out = self.layer1(out) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        
        if self.use_final:
            out = self.fc(out)

        return out

    def _layer(self, block, num_layers, stride, out_channels):
        '''
        making a residual layer containing num_layers number of residual blocks. 
        '''
        layers = []
        downsampling_layer = None # the downsampling will be filled if it matched the condition of:
            # the stride being greater than one or twe want to downsample for identity mapping
        if stride != 1 or self.in_channels != out_channels * 1:
            downsampling_layer = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
        
        # add the block for downsampling 
        layers.append(block(self.in_channels, out_channels, downsampling_layer, stride))
        self.in_channels = out_channels * self.expansion # updating for passing it to the next layer of the block

        # adding blocks
        for _ in range(num_layers - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
