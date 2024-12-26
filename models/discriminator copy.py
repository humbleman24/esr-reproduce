import torch
import torch.nn as nn
import torchvision.models as models

class Discriminator(nn.Module):
    def __init__(self, pretrained=True, requires_grad=False):
        super(Discriminator, self).__init__()
        # 加载预训练的 VGG16 模型
        vgg = models.vgg16(pretrained=pretrained)
        # 提取特征部分（去除最后的全连接层）
        self.features = vgg.features
        # 根据需要添加自定义的判别层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
        
        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x