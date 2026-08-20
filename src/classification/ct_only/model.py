"""
ResNet-50 model setup for the CT vascular-contact classification task.

Supports both 2D (single-slice, replicated to 3 channels) and 2.5D
(5-slice stack, used directly as 5 input channels) input formats.

Due to the small dataset and class imbalance (16% positive class), a pretrained
resnet-50 is used.

"""

import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet50(input_channels=3, num_classes=2, freeze_until="layer4"):
    ## The first conv layer is replaced to accept the input channel count used here
    # (3 for 2D, 5 for 2.5D) since the pretrained layer expects 3-channel RGB.

    model = models.resnet50(weights='IMAGENET1K_V2')

    if input_channels != 3:
        old_conv1 = model.conv1
        new_conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=(old_conv1.bias is not None),
        )
        with torch.no_grad():
            avg_weight = old_conv1.weight.mean(dim=1, keepdim=True)  # (out_ch, 1, k, k)
            new_conv1.weight[:] = avg_weight.repeat(1, input_channels, 1, 1)
        model.conv1 = new_conv1

    # Added dropout before the final linear layer as a regularization measure,
    # given the overfitting observed in the first training run (train loss
    # dropped to ~0.08 while val loss rose to ~1.8 over 19 epochs).
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )

    if freeze_until == "all":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    elif freeze_until == "layer4":
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    # added layer3_layer4 because it gave better results when testing
    elif freeze_until == "layer3_layer4":
        for name, param in model.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    elif freeze_until == "none":
        pass  # everything trainable

    else:
        raise ValueError(f"Unknown freeze_until option: {freeze_until}")

    return model


def prepare_2d_batch(batch_1channel):
    return batch_1channel.repeat(1, 3, 1, 1)


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    print("=== Testing 2D model (3-channel input) ===")
    model_2d = build_resnet50(input_channels=3, num_classes=2, freeze_until="layer4")
    trainable, total = count_trainable_params(model_2d)
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # test forward pass with a dummy batch
    dummy_1channel = torch.rand(4, 1, 224, 224)
    dummy_3channel = prepare_2d_batch(dummy_1channel)
    print(f"Dummy input shape: {dummy_3channel.shape}")
    output = model_2d(dummy_3channel)
    print(f"Output shape: {output.shape}  (should be [4, 2])")

    print("\n=== Testing 2.5D model (5-channel input) ===")
    model_2_5d = build_resnet50(input_channels=5, num_classes=2, freeze_until="layer4")
    trainable, total = count_trainable_params(model_2_5d)
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    dummy_5channel = torch.rand(4, 5, 224, 224)
    print(f"Dummy input shape: {dummy_5channel.shape}")
    output = model_2_5d(dummy_5channel)
    print(f"Output shape: {output.shape}  (should be [4, 2])")


# === Testing 2D model (3-channel input) ===
# Trainable params: 14,968,834 / 23,512,130 (63.7%)
# Dummy input shape: torch.Size([4, 3, 224, 224])
# Output shape: torch.Size([4, 2])  (should be [4, 2])

# === Testing 2.5D model (5-channel input) ===
# Trainable params: 14,968,834 / 23,518,402 (63.6%)
# Dummy input shape: torch.Size([4, 5, 224, 224])
# Output shape: torch.Size([4, 2])  (should be [4, 2])
