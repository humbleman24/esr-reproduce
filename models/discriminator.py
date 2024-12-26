import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, 256, 256)):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        # Initial block
        self.model = nn.Sequential(
            conv_block(channels, 64, kernel_size=3, stride=1, padding=1, use_batchnorm=False),
            conv_block(64, 64, kernel_size=3, stride=2, padding=1),
            conv_block(64, 128, kernel_size=3, stride=1, padding=1),
            conv_block(128, 128, kernel_size=3, stride=2, padding=1),
            conv_block(128, 256, kernel_size=3, stride=1, padding=1),
            conv_block(256, 256, kernel_size=3, stride=2, padding=1),
            conv_block(256, 512, kernel_size=3, stride=1, padding=1),
            conv_block(512, 512, kernel_size=3, stride=2, padding=1),
        )
        # Fully connected layers
        ds_size = height // 2**4
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size * ds_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity
