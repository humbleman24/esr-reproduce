import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from models.losses import GeneratorLoss, DiscriminatorLoss
from utils.dataset import SRDataset
from utils.logger import Logger
import torchvision.transforms as transforms
from torchvision.models import vgg19

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 加载配置
    config = load_config('config/config.yaml')

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 加载数据集
    train_dataset = SRDataset(
        lr_dir=config['train']['dataset']['lr_dir'],
        hr_dir=config['train']['dataset']['hr_dir'],
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers']
    )

    # 初始化模型
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 定义优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=float(config['train']['learning_rate']), betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=float(config['train']['learning_rate']), betas=(0.9, 0.999))

    # 定义损失函数
    # 使用预训练的 VGG19 作为特征提取器
    vgg = vgg19(pretrained=True).features[:35].to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    criterion_G = GeneratorLoss(feature_extractor=vgg).to(device)
    criterion_D = DiscriminatorLoss().to(device)

    # 日志记录
    logger = Logger(log_dir=config['train']['logs_dir'])

    # 训练循环
    for epoch in range(1, config['train']['num_epochs'] + 1):
        generator.train()
        discriminator.train()
        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device)
            hr = hr.to(device)

            # ---------------------
            # 训练判别器
            # ---------------------
            optimizer_D.zero_grad()
            fake_hr = generator(lr).detach()
            valid = discriminator(hr)
            fake = discriminator(fake_hr)
            loss_D = criterion_D(valid, fake)
            loss_D.backward()
            optimizer_D.step()

            # ---------------------
            # 训练生成器
            # ---------------------
            optimizer_G.zero_grad()
            fake_hr = generator(lr)
            validity = discriminator(fake_hr)
            loss_G, adv_loss, content_loss = criterion_G(fake_hr, hr)
            loss_G.backward()
            optimizer_G.step()

            # 日志记录
            logger.log('Loss/Generator', loss_G.item(), epoch)
            logger.log('Loss/Discriminator', loss_D.item(), epoch)
            logger.log('Loss/Adversarial', adv_loss.item(), epoch)
            logger.log('Loss/Content', content_loss.item(), epoch)

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{config['train']['num_epochs']}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

        # 每隔一定周期保存模型
        if epoch % config['train']['save_model_every'] == 0:
            checkpoint_path = os.path.join(config['train']['checkpoint_dir'], f'generator_epoch_{epoch}.pth')
            torch.save(generator.state_dict(), checkpoint_path)
            print(f"保存生成器模型到 {checkpoint_path}")

    # 训练完成后保存最终模型
    torch.save(generator.state_dict(), os.path.join(config['train']['checkpoint_dir'], 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config['train']['checkpoint_dir'], 'discriminator_final.pth'))
    logger.close()
    print("训练完成！")

if __name__ == "__main__":
    main()
