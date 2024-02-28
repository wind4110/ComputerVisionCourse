import argparse
import torch
from torchvision import transforms
import torch.nn.functional as F
from unet import UNet
import matplotlib.pyplot as plt
from PIL import Image


def plot_img_and_mask(img, mask):
    classes = mask.max()
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Mask')
    ax[1].imshow(mask == 0)
    plt.savefig("a.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict masks from input images')
    parser.add_argument('--model',
                        '-m',
                        default='model.pth',
                        help='Specify the file in which the model is stored')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading model {args.model}')
    print(f'Using device {device}')

    model = UNet(in_channels=3, out_channels=1).to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)

    print('Model loaded')

    # 输入图片路径
    input_path = 'ComputerVision/HW4_LearningCNN/infer.jpg'
    # 载入图片
    input_x = Image.open(input_path)
    img = input_x

    # 定义预处理
    transform = transforms.Compose(
        [transforms.Resize([572, 572]),
         transforms.ToTensor()])
    # 调整大小为572×572，转为张量，增加维度B
    input_x_tensor = transform(input_x).unsqueeze(0)  # x.shape = (1, 3, 572, 572)

    # 测试
    with torch.no_grad():
        input_x_tensor = input_x_tensor.to(device)
        score = model(input_x_tensor)

    # score.shape = (1, 1, w, h)
    # 输出结果插值处理，匹配图片尺寸
    img_size = (img.size[1], img.size[0])
    score = F.interpolate(score,
                          size=img_size,
                          mode="nearest")
    score = score.squeeze()  # 移除尺度为1的维度, score.shape = (w, h)
    score = F.sigmoid(score)  # sigmoid函数处理
    threshold = 0.8  # 设定阈值，不能超过1，过小效果变差
    mask = (score > threshold)  # 根据阈值转换
    mask = mask.cpu().numpy().astype(int)  # 张量迁移至cpu，转为整数数组

    plot_img_and_mask(img, mask)  # 输出结果图片

    print("Finished.")
