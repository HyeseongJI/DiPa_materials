import torch
import segmentation_models_pytorch as smp

def get_model(model_name, classes=1, encoder_weights=None):
    """
    Returns the specific model architecture.
    
    Args:
        model_name (str): One of ['mobilenet', 'resnet18', 'resnet50', 'segformer']
        classes (int): Number of output classes (default: 1 for binary segmentation)
        encoder_weights (str or None): Pre-trained weights name (e.g., 'imagenet') or None.
    """
    model_name = model_name.lower()
    
    print(f"🔥 Initializing Model: {model_name.upper()}")

    # 1. MobileNetV3-Large (Proposed Method)
    if model_name == "mobilenet":
        return smp.Unet(
            encoder_name="timm-mobilenetv3_large_100",
            encoder_weights=encoder_weights, 
            in_channels=3,
            classes=classes,
            activation="sigmoid"
        )

    # 2. ResNet-18 (Comparison)
    elif model_name == "resnet18":
        return smp.Unet(
            encoder_name="resnet18",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation="sigmoid"
        )

    # 3. ResNet-50 (Comparison)
    elif model_name == "resnet50":
        return smp.Unet(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation="sigmoid"
        )

    # 4. SegFormer / MiT-B1 (Transformer Comparison)
    elif model_name == "segformer":
        return smp.Unet(
            encoder_name="mit_b1",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation="sigmoid"
        )
        
    else:
        raise ValueError(f"❌ Unknown model name: {model_name}. Choose from [mobilenet, resnet18, resnet50, segformer]")