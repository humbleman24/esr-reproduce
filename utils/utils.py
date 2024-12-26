import os
import cv2

def downsample_image(input_dir, output_dir, scale=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        lr = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_dir, img_name), lr)

if __name__ == "__main__":
    import yaml

    # 读取配置文件
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_hr_dir = config['train']['dataset']['hr_dir']
    train_lr_dir = config['train']['dataset']['lr_dir']
    valid_hr_dir = config['train']['dataset']['valid_hr_dir']
    valid_lr_dir = config['train']['dataset']['valid_lr_dir']
    scale = config['train']['dataset']['scale']

    downsample_image(train_hr_dir, train_lr_dir, scale)
    downsample_image(valid_hr_dir, valid_lr_dir, scale)

    print("低分辨率图像生成完成！")
