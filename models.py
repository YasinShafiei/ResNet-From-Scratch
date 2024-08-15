'''
Use of ResNet models written in architecture.py
'''
# import libraries
import architectures
import torch

def ResNet18(img_channels, num_classes, device='cuda', use_final=True):
    '''
    Using the architectures to build a resnet18 model
    '''
    # raising an error for not entering true parameters to the function
    if use_final and num_classes is None:
        raise Exception('when use_final is set to true, it is necessary to give an integer value to num_classes')
    
    # define the model under different conditions
    model = architectures.SmallResNet(
        num_layers=[2, 2, 2, 2],
        img_channels=img_channels,
        num_classes=num_classes if use_final else None,
        use_final=use_final
    ).to(device=device)

    return model

def ResNet32(img_channels, num_classes, device='cuda', use_final=True):
    '''
    Using the architectures to build a resnet32 model
    '''
    # raising an error for not entering true parameters to the function
    if use_final and num_classes is None:
        raise Exception('when use_final is set to true, it is necessary to give an integer value to num_classes')
    
    # define the model under different conditions
    model = architectures.SmallResNet(
        num_layers=[3, 4, 6, 3],
        img_channels=img_channels,
        num_classes=num_classes if use_final else None,
        use_final=use_final
    ).to(device=device)

    return model

def ResNet50(img_channels, num_classes, device='cuda', use_final=True):
    '''
    Using the architectures to build a resnet50 model
    '''
    # raising an error for not entering true parameters to the function
    if use_final and num_classes is None:
        raise Exception('when use_final is set to true, it is necessary to give an integer value to num_classes')
    
    # define the model under different conditions
    model = architectures.LargeResNet(
        num_blocks=[3, 4, 6, 3],
        img_channels=img_channels,
        num_classes=num_classes if use_final else None,
        use_final=use_final
    ).to(device=device)

    return model

def ResNet101(img_channels, num_classes, device='cuda', use_final=True):
    '''
    Using the architectures to build a resnet101 model
    '''
    # raising an error for not entering true parameters to the function
    if use_final and num_classes is None:
        raise Exception('when use_final is set to true, it is necessary to give an integer value to num_classes')
    
    # define the model under different conditions
    model = architectures.LargeResNet(
        num_blocks=[3, 4, 23, 3],
        img_channels=img_channels,
        num_classes=num_classes if use_final else None,
        use_final=use_final
    ).to(device=device)

    return model

def ResNet152(img_channels, num_classes, device='cuda', use_final=True):
    '''
    Using the architectures to build a resnet152 model
    '''
    # raising an error for not entering true parameters to the function
    if use_final and num_classes is None:
        raise Exception('when use_final is set to true, it is necessary to give an integer value to num_classes')
    
    # define the model under different conditions
    model = architectures.LargeResNet(
        num_blocks=[3, 8, 36, 3],
        img_channels=img_channels,
        num_classes=num_classes if use_final else None,
        use_final=use_final
    ).to(device=device)

    return model
