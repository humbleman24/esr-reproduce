import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorLoss(nn.Module):
    def __init__(self, feature_extractor, lambda_adv=1e-3, lambda_content=1):
        super(GeneratorLoss, self).__init__()
        self.adversarial_loss = nn.BCELoss()
        self.content_loss = nn.MSELoss()
        self.feature_extractor = feature_extractor
        self.lambda_adv = lambda_adv
        self.lambda_content = lambda_content

    def forward(self, fake, real):
        # Adversarial loss
        validity = self.feature_extractor(fake)
        adversarial_loss = self.adversarial_loss(validity, torch.ones_like(validity).to(fake.device))
        # Content loss
        real_features = self.feature_extractor(real).detach()
        fake_features = self.feature_extractor(fake)
        content_loss = self.content_loss(fake_features, real_features)
        # Total loss
        total_loss = self.lambda_content * content_loss + self.lambda_adv * adversarial_loss
        return total_loss, adversarial_loss, content_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.adv_loss = nn.BCELoss()

    def forward(self, valid, fake):
        valid_loss = self.adv_loss(valid, torch.ones_like(valid).to(valid.device))
        fake_loss = self.adv_loss(fake, torch.zeros_like(fake).to(fake.device))
        total_loss = (valid_loss + fake_loss) / 2
        return total_loss
