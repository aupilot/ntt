import torch

from torchvision.models.resnet import BasicBlock
import torch.nn as nn
import torch.nn.functional as F
import math


# change the layer order within resnet block
class NewBlock2(BasicBlock):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(NewBlock2, self).__init__(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample)
        self.bn0 = nn.BatchNorm2d(inplanes)
        del self.bn2

    def forward(self, x):
        residual = x

        out = self.bn0(x)
        out = F.leaky_relu(out)
        out = self.conv1(out)

        out = self.bn1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class ResNetLight(nn.Module):

    def __init__(self, block, layers):
        self.name = "ResNetSkip"
        self.inplanes = 64
        super(ResNetLight, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=2, bias=False)
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer4 = nn.Conv2d(256, 512, kernel_size=(8,1), stride=1, padding=0, bias=False)
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        self.avgpool = nn.AvgPool2d((1,8), stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.drop1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(896, 1024)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(1024, 6)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)      #128,8,8
        x4 = self.layer4(x3)

        skip2 = F.avg_pool2d(x2, (16, 3), stride=2, padding=(0, 1))
        skip3 = F.avg_pool2d(x3, (8,3), stride=1, padding=(0,1))
        xx = torch.cat([x4, skip3, skip2], 1)

        x = self.avgpool(xx)
        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.logsoftmax(x)

        return x


class ResNetVeryLight(nn.Module):

    def __init__(self, block, layers):
        self.name = "ResNetSkip"
        self.inplanes = 32
        super(ResNetVeryLight, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2, bias=False)
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer4 = nn.Conv2d(128, 256, kernel_size=(8,1), stride=1, padding=0, bias=False)
        # self.avgpool = nn.AvgPool2d((1,8), stride=1)
        self.avgpool = nn.MaxPool2d((1,8), stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.drop1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(448, 512)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(512, 6)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)      #128,8,8
        x4 = self.layer4(x3)

        skip2 = F.avg_pool2d(x2, (16, 3), stride=2, padding=(0, 1))
        skip3 = F.avg_pool2d(x3, (8,3), stride=1, padding=(0,1))
        xx = torch.cat([x4, skip3, skip2], 1)

        x = self.avgpool(xx)
        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.logsoftmax(x)

        return x


class ResNetB(nn.Module):

    def __init__(self, block, layers):
        self.name = "ResNetSkip"
        self.inplanes = 32
        super(ResNetB, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = nn.Conv2d(128, 256, kernel_size=(5,5), stride=1, padding=0, bias=False)
        self.avgpool = nn.MaxPool2d((4,4), stride=0)
        self.drop1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(448, 512)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(512, 6)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)      #128,8,8
        x4 = self.layer4(x3)

        skip2 = F.avg_pool2d(x2, (7, 7), stride=3, padding=(0, 0))
        skip3 = F.avg_pool2d(x3, (3, 3), stride=2, padding=(1, 1))
        xx = torch.cat([x4, skip3, skip2], 1)

        x = self.avgpool(xx)
        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.logsoftmax(x)

        return x


def resnet_light(pretrained=False, **kwargs):
    """Constructs a ResNet model with 1-channel input and small number of layers.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetLight(BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet_light2(pretrained=False, **kwargs):
    """Constructs a ResNet model with 1-channel input and small number of layers.
    Args:
        pretrained (bool): n/a
    """
    model = ResNetLight(NewBlock2, [2, 2, 2, 2], **kwargs)
    return model


def resnet_vlight(pretrained=False, **kwargs):
    """Constructs a ResNet model with 1-channel input and small number of layers.
    Args:
        pretrained (bool): n/a
    """
    model = ResNetVeryLight(NewBlock2, [2, 2, 2, 2], **kwargs)
    return model

def resnet_b(pretrained=False, **kwargs):
    """Constructs a ResNet model with 1-channel input and small number of layers.
    Args:
        pretrained (bool): n/a
    """
    model = ResNetB(NewBlock2, [2, 2, 3, 3], **kwargs)
    return model
