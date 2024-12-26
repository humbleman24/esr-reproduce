import os
import yaml
import torch
from models.generator import Generator
from utils.dataset import SRDataset
from torchvision.transforms import ToTensor
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_image(tensor, path):
    image = tensor.clamp(0, 1).cpu().detach().numpy()
    image = image.squeeze().transpose(1, 2, 0) * 255
    image = Image.fromarray(image.astype('uint8'))
    image.save(path)

def main():
    # 加载配置
    config = load_config('config/config.yaml')

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化生成器
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(config['test']['model_path'], map_location=device))
    generator.eval()

    # 定义转换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 创建输出目录
    os.makedirs(config['test']['output_dir'], exist_ok=True)

    # 获取测试图像列表
    test_images = sorted(os.listdir(config['test']['input_dir']))

    for img_name in tqdm(test_images, desc="处理图像"):
        img_path = os.path.join(config['test']['input_dir'], img_name)
        lr_image = load_image(img_path, transform).to(device)
        with torch.no_grad():
            sr_image = generator(lr_image)
        save_path = os.path.join(config['test']['output_dir'], img_name)
        save_image(sr_image, save_path)

    print("推理完成！高分辨率图像保存在", config['test']['output_dir'])

if __name__ == "__main__":
    main()
