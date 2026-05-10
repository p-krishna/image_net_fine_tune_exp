# model.py — Model factory: loads pretrained model, freezes backbone, replaces head

import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES


def build_model( model_name: str ) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "alexnet":
        # initialize alexnet with pretrained weights for all layers
        model = models.alexnet( weights=models.AlexNet_Weights.DEFAULT )
        # freeze all layers by asking pytorch not to compute gradients for them
        _freeze_backbone( model )
        # replace the final classifier layer with a new one that has NUM_CLASSES outputs
        model.classifier[6] = nn.Linear( 4096, NUM_CLASSES )

    elif model_name == "vgg16":
        model = models.vgg16( weights=models.VGG16_Weights.DEFAULT )
        _freeze_backbone( model )
        model.classifier[6] = nn.Linear( 4096, NUM_CLASSES )

    elif model_name == "resnet50":
        model = models.resnet50( weights=models.ResNet50_Weights.DEFAULT )
        _freeze_backbone( model )
        model.fc = nn.Linear( model.fc.in_features, NUM_CLASSES )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0( weights=models.EfficientNet_B0_Weights.DEFAULT )
        _freeze_backbone( model )
        model.classifier[1] = nn.Linear( model.classifier[1].in_features, NUM_CLASSES )

    else:
        raise ValueError( f"Unknown model: {model_name}" )

    return model


def _freeze_backbone( model: nn.Module ):
    for param in model.parameters():
        param.requires_grad = False


def count_trainable_params( model: nn.Module ) -> int:
    return sum( p.numel() for p in model.parameters() if p.requires_grad )