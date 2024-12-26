import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)    # return to nf dimension as output

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        output_conv1 = self.lrelu(self.conv1(x))
        output_conv2 = self.lrelu(self.conv2(torch.cat((x, output_conv1), 1)))   # 1 is the dimension to concat
        output_conv3 = self.lrelu(self.conv3(torch.cat((x, output_conv1, output_conv2), 1)))
        output_conv4 = self.lrelu(self.conv4(torch.cat((x, output_conv1, output_conv2, output_conv3), 1)))
        output_conv5 = self.conv5(torch.cat((x, output_conv1, output_conv2, output_conv3, output_conv4), 1))
        return output_conv5 * 0.2 + x
    
class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()

        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        output = self.RDB1(x)
        output = self.RDB2(output)
        output = self.RDB3(output)
        return output * 0.2 + x
    
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()

        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = self.make_layer(RRDB_block_f, nb)
         
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)       # following the structure of RRDB, it would add the output of first conv output, so the output layer is nf

        # upsampling layers, learning the interpolation
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def make_layer(self, block, n_blocks):
        layers = []
        for _ in range(n_blocks):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        output1 = self.conv_first(x)
        output_rrdb = self.RRDB_trunk(output1)
        output = self.trunk_conv(output_rrdb)
        output = output + output1

        # interpolation would not change the channel size
        # so the upconv layer should have the same channel size as the input
        # try bilinear interpolation next time
        output = self.lrelu(self.upconv1(F.interpolate(output, scale_factor=2, mode='nearest')))
        output = self.lrelu(self.upconv2(F.interpolate(output, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(output)))
        return out



