import torch
import torch.nn as nn
import torch.nn.functional as F

class MyResidualBlock(nn.Module):
    def __init__(self, down_sample):
        super(MyResidualBlock, self).__init__()
        self.down_sample = down_sample
        self.stride = 2 if self.down_sample else 1
        K = 9
        P = (K-1) // 2
        self.conv1 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, K),
                               stride=(1, self.stride),
                               padding=(0, P),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, K),
                               padding=(0, P),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        if self.down_sample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.idfunc_1 = nn.Conv2d(in_channels=256,
                                      out_channels=256,
                                      kernel_size=(1, 1),
                                      bias=False)

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.down_sample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)

        if identity.size(2) != x.size(2) or identity.size(3) != x.size(3):
            diff = x.size(3) - identity.size(3)
            if diff > 0:  # x is larger
                identity = F.pad(identity, (0, diff), "constant", 0)
            else:  # identity is larger, need to crop or pool (example uses cropping)
                identity = identity[:, :, :, :x.size(3)]

        x = x+identity
        return x


class NN(nn.Module):
    def __init__(self, nOUT):
        super(NN, self).__init__()
        self.conv = nn.Conv2d(in_channels=12,
                              out_channels=256,
                              kernel_size=(1, 15),
                              padding=(0, 7),
                              stride=(1, 2),
                              bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.rb_0 = MyResidualBlock(down_sample=True)
        self.rb_1 = MyResidualBlock(down_sample=True)
        self.rb_2 = MyResidualBlock(down_sample=True)
        self.rb_3 = MyResidualBlock(down_sample=True)
        self.rb_4 = MyResidualBlock(down_sample=True)

        self.mha = nn.MultiheadAttention(256, 8)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.fc_1 = nn.Linear(256 + 12, nOUT)

        self.ch_fc1 = nn.Linear(nOUT, 256)
        self.ch_bn = nn.BatchNorm1d(256)
        self.ch_fc2 = nn.Linear(256, nOUT)

    def forward(self, x, leads):
        x = F.leaky_relu(self.bn(self.conv(x)))

        x = self.rb_0(x)
        x = self.rb_1(x)
        x = self.rb_2(x)
        x = self.rb_3(x)
        x = self.rb_4(x)

        x = F.dropout(x, p=0.5, training=self.training)

        x = x.squeeze(2).permute(2,0,1)
        x,s = self.mha(x, x, x)
        x = x.permute(1, 2, 0)
        x = self.pool(x).squeeze(2)
        x = torch.cat((x, leads), dim=1)

        x = self.fc_1(x)
        p = x.detach()
        p = F.leaky_relu(self.ch_bn(self.ch_fc1(p)))
        p = torch.sigmoid(self.ch_fc2(p))
        return x, p


if __name__ == '__main__':
    import torch

    x = torch.randn(64, 12, 1, 5000)
    l = torch.ones(64, 12)
    model = NN(50)
    y, p = model(x, l)
    print()
