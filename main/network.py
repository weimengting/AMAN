import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class AMAN(nn.Module):
    def __init__(self, num_classes):
        '''alphas and betas are attention scores for weighting amplification factors and frames respectively'''
        super(AMAN, self).__init__()

        self.inplanes = 256
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.alpha = nn.Sequential(nn.Linear(512, 1),
                                   nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(512, 1),
                                  nn.Sigmoid())
        self.pred_fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        attentions = []
        vms = []

        num_pair = x.size(2)  # 9
        for j in range(x.size(1)): # 11
            alphas = []
            vs = []
            for i in range(num_pair):
                f = x[:, j, i, :, :, :]
                f = self.layer4(f)
                f = self.avgpool(f)
                f = f.squeeze(2).squeeze(2)
                vs.append(f)
                alphas.append(self.alpha(f))
            alphas_stack = torch.stack(alphas, dim=2)
            vs_stack = torch.stack(vs, dim=2)
            vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
            vms.append(vm1)
            attentions.append(self.beta(vm1))

        attentions_stack = torch.stack(attentions, dim=0)
        vm_stack = torch.stack(vms, dim=0)
        vm2 = vm_stack.mul(attentions_stack).sum(0).div(attentions_stack.sum(0))
        pred = self.pred_fc(vm2)
        return pred



if __name__ == '__main__':
    ri = AMAN(num_classes=5).cuda()
    x = torch.randn((4, 11, 9, 256, 14, 14)).cuda()
    pred = ri.forward(x)
    print(pred.shape)