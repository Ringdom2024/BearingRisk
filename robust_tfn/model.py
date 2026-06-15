import torch
from torch import nn

from Models.BackboneCNN import CNN
from Models.TFN import TFN_STTF


class CounterfactualTFN(nn.Module):
    """TFN-STTF wrapper exposing channel responses for frequency interventions."""

    def __init__(
        self,
        mid_channel=32,
        num_classes=10,
        sample_rate=48_000.0,
        fault_ratios=None,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.fault_ratios = dict(fault_ratios or {})
        self.backbone = TFN_STTF(
            in_channels=1,
            out_channels=num_classes,
            mid_channel=mid_channel,
            clamp_flag=True,
        )

    @property
    def center_frequencies(self):
        return self.backbone.funconv.superparams[:, 0, 0]

    def classify_tf(self, tf_response):
        x = self.backbone.layer1(tf_response)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = x.view(x.size(0), -1)
        features = self.backbone.layer5(x)
        return self.backbone.fc(features), features

    def forward(self, inputs, channel_mask=None, return_details=False):
        tf_response = self.backbone.funconv(inputs)
        if channel_mask is not None:
            tf_response = tf_response * channel_mask.unsqueeze(-1)
        logits, features = self.classify_tf(tf_response)
        if return_details:
            return {
                "logits": logits,
                "features": features,
                "tf_response": tf_response,
                "channel_energy": tf_response.mean(dim=-1),
            }
        return logits


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.activation(self.main(inputs) + self.skip(inputs))


class ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        channels = [32, 64, 128, 256]
        stages = []
        in_channels = 32
        for stage_index, out_channels in enumerate(channels):
            stride = 1 if stage_index == 0 else 2
            stages.extend(
                [
                    ResidualBlock1D(in_channels, out_channels, stride),
                    ResidualBlock1D(out_channels, out_channels),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, inputs):
        features = self.features(self.stem(inputs))
        return self.classifier(self.pool(features).squeeze(-1))


def build_baseline_model(name, num_classes):
    if name == "CNN":
        return CNN(in_channels=1, out_channels=num_classes)
    if name == "ResNet1D":
        return ResNet1D(num_classes)
    raise ValueError(f"Unknown baseline model: {name}")
