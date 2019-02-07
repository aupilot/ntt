import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

from torchviz import make_dot


class NewBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(NewBlock2, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.BatchNorm1d(in_planes),
                # nn.LeakyReLU(inplace=True),
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out += self.shortcut(x)
        return out


# skip connections for the second last 2 layers
class SuperNet72(nn.Module):
    def __init__(self, block, input_depth, num_blocks, num_classes=6, name='No Name', init_with_ae=None):
        super(SuperNet72, self).__init__()
        self.name = name

        self.inplanes_before_cat = input_depth
        self.num_filters = 72
        self.in_planes = self.num_filters                                   # N low pass filters constructed below
        # self.in_planes = self.inplanes_before_cat * self.num_filters       # N low pass filters constructed below
        self.lowpass_size = 15 #17 #21 #31 # 25 # 31 # 51 # 81

        # N low pass filters
        self.conv1 = nn.Conv1d(input_depth, input_depth*self.num_filters,
                               kernel_size=self.lowpass_size,
                               stride=1,
                               padding=0,#(self.lowpass_size - 1) / 2,
                               groups=input_depth,
                               bias=True)

        self.conv2 = nn.Conv1d(input_depth*self.num_filters, self.in_planes,
                               kernel_size=9,
                               stride=3,
                               padding=0,#2,
                               groups=1, #input_depth,
                               bias=True)

        self.bn2 = nn.BatchNorm1d(self.in_planes)       # this must be N * in_planes_before_cat
        self.layer1 = self._make_layer(block, 128,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 192, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 768, num_blocks[4], stride=2)
        self.linear = nn.Linear(1536, num_classes)
        # nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.weight,0)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        if init_with_ae is not None:
            self.init_with_auto(init_with_ae)

    def init_with_auto(self, trained_ae):
        resnet_ae = torch.load(trained_ae, map_location='cuda:0')
        try:
            # self.conv1.weight = torch.nn.Parameter(resnet_ae.resnet.conv1.weight.to('cuda'))
            # self.conv1.weight = torch.nn.Parameter(resnet_ae.resnet.conv1.weight.to('cuda'))
            self.load_state_dict(resnet_ae.resnet.state_dict())
        except:
            print('Make sure that the number of channels at AutoEncoder and ResNet are the same!)')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = torch.log_(1+torch.abs_(out))
        out = self.conv2(out)

        out = F.leaky_relu(self.bn2(out))

        out = F.avg_pool1d(out, kernel_size=7, padding=0, stride=3)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

#        skip1 = F.avg_pool1d(out1, 15, stride=15)
#        skip2 = F.avg_pool1d(out2, 7, stride=7)
        skip3 = F.avg_pool1d(out3, 4, stride=4, padding=1)
        skip4 = F.avg_pool1d(out4, 3, stride=2, padding=1)

        out = torch.cat([out5, skip4, skip3], 1)

        out = out[:,:,1:-1]             # обрезаем концы, чтобы padding не вызывал глюков

        out = F.avg_pool1d(out, 26)
        out = out.view(out.size(0), -1)
        out = self.linear(F.dropout(out, training=self.training, p=0.25))
        out = self.logsoftmax(out)
        return out


def SuperNet740(input_depth=1, init_with_ae=None):
    # this net has larger receptive field comparing to standard resnet
    return SuperNet72(NewBlock2, input_depth, [2, 2, 2, 2, 2],
                      num_classes=6,
                      name='SuperNet740 - 72x15 filters, 2-2-2-2-2 layers, skips 3&4, new Block2',
                      init_with_ae=None)


def test():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net = SuperNet740(input_depth=1)
    net.cuda()
    x = torch.randn(2,1,4000)
    y = net(x)
    print(y.size())
    summary(net, (1,4000))
    g=make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
    g.view()


if __name__ == '__main__':
    test()

