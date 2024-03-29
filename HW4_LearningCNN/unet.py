import torch
# from torchvision.transforms.functional import center_crop
from torch import nn
from torchvision.transforms.functional import center_crop


class CropAndConcat(nn.Module):
    """
    ### Crop and Concatenate the feature map

    Crop the feature map from the contracting path to the size of the current feature map
    """

    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """

        b, c, h, w = x.shape

        # TODO: Concatenate the feature maps
        # use torchvision.transforms.functional.center_crop(...)
        x = torch.cat((x, center_crop(contracting_x, (h, w))), dim=1)

        return x


class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # TODO: Double convolution layers for the contracting path.
        # Number of features gets doubled at each step starting from 64.
        num = 64
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      num,
                      kernel_size=3,
                      padding=1,
                      stride=1), nn.ReLU(),
            nn.Conv2d(num, num, kernel_size=3, padding=1, stride=1), nn.ReLU())

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(num, num * 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(num * 2, num * 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU())

        self.down_conv3 = nn.Sequential(
            nn.Conv2d(num * 2, num * 4, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(num * 4, num * 4, kernel_size=3, padding=1, stride=1),
            nn.ReLU())

        self.down_conv4 = nn.Sequential(
            nn.Conv2d(num * 4, num * 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(num * 8, num * 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU())

        # Down sampling layers for the contracting path
        self.down_sample1 = nn.MaxPool2d(2)
        self.down_sample2 = nn.MaxPool2d(2)
        self.down_sample3 = nn.MaxPool2d(2)
        self.down_sample4 = nn.MaxPool2d(2)

        # TODO: The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = nn.Sequential(
            nn.Conv2d(num * 8, num * 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(num * 16, num * 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU())

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        num = 1024
        self.up_sample1 = nn.ConvTranspose2d(num,
                                             num // 2,
                                             kernel_size=2,
                                             stride=2)
        self.up_sample2 = nn.ConvTranspose2d(num // 2,
                                             num // 4,
                                             kernel_size=2,
                                             stride=2)
        self.up_sample3 = nn.ConvTranspose2d(num // 4,
                                             num // 8,
                                             kernel_size=2,
                                             stride=2)
        self.up_sample4 = nn.ConvTranspose2d(num // 8,
                                             num // 16,
                                             kernel_size=2,
                                             stride=2)

        # TODO: Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the contracting path.
        # Therefore, the number of input features is double the number of features from up-sampling.
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(num, num // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(num // 2, num // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU())

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(num // 2, num // 4, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(num // 4, num // 4, kernel_size=3, padding=1, stride=1),
            nn.ReLU())

        self.up_conv3 = nn.Sequential(
            nn.Conv2d(num // 4, num // 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(num // 8, num // 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU())

        self.up_conv4 = nn.Sequential(
            nn.Conv2d(num // 8, num // 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(num // 16, num // 16, kernel_size=3, padding=1,
                      stride=1), nn.ReLU())

        # Crop and concatenate layers for the expansive path.
        # TODO: Implement class CropAndConcat starting from line 6
        self.concat1 = CropAndConcat()
        self.concat2 = CropAndConcat()
        self.concat3 = CropAndConcat()
        self.concat4 = CropAndConcat()

        # TODO: Final 1*1 convolution layer to produce the output
        self.final_conv = nn.Conv2d(num // 16,
                                    1,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: input image
        """
        # TODO: Contracting path
        # Remember to pass middle result to the expansive path
        x = self.down_conv1(x)
        contracting_x1 = x
        x = self.down_sample1(x)
        x = self.down_conv2(x)
        contracting_x2 = x
        x = self.down_sample2(x)
        x = self.down_conv3(x)
        contracting_x3 = x
        x = self.down_sample3(x)
        x = self.down_conv4(x)
        contracting_x4 = x
        x = self.down_sample4(x)
        # ...

        # Two 3*3 convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)

        # TODO: Expansive path
        # Remember to receive from contracting path and concat
        x = self.up_sample1(x)
        x = self.concat1.forward(x, contracting_x4)
        x = self.up_conv1(x)

        x = self.up_sample2(x)
        x = self.concat2.forward(x, contracting_x3)
        x = self.up_conv2(x)

        x = self.up_sample3(x)
        x = self.concat3.forward(x, contracting_x2)
        x = self.up_conv3(x)

        x = self.up_sample4(x)
        x = self.concat4.forward(x, contracting_x1)
        x = self.up_conv4(x)

        # Final 1*1 convolution layer
        x = self.final_conv(x)

        return x
