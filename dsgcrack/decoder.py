"""Multilevel fusion decoder containing the SRG module."""
import torch
from torch import nn
from .srg import SRG

class Conv2d_GN(nn.Module):
    def __init__(self, inc, outc, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bias=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, ks, stride, pad, dilation, groups, bias),
            nn.GroupNorm(outc//8, outc)
        )
    def forward(self, x):
        return self.block(x)


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768, identity=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        if identity:
            self.proj = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.permute(0,2,1).reshape(n, -1, h, w)
        
        return x


class MultiScaleDecoder(nn.Module):
    def __init__(self, embed_dims, mid_dim=8, size=(512,512), ):
        super(MultiScaleDecoder, self).__init__()
        
        self.size = size

        self.mlp1 = MLP(input_dim=embed_dims[0], embed_dim=mid_dim)
        self.mlp2 = MLP(input_dim=embed_dims[1], embed_dim=mid_dim)
        self.mlp3 = MLP(input_dim=embed_dims[2], embed_dim=mid_dim)
        self.mlp4 = MLP(input_dim=embed_dims[3], embed_dim=mid_dim)

        self.conv1 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv2 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv3 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv4 = Conv2d_GN(inc=mid_dim, outc=mid_dim)



        self.srg = SRG(dim=mid_dim)

        self.block = nn.Sequential(
            Conv2d_GN(inc=mid_dim*4, outc=mid_dim*4),
            Conv2d_GN(inc=mid_dim*4, outc=mid_dim),
            nn.ReLU()
        )


        self.linear_pred = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim//4, 1),
            nn.Conv2d(mid_dim//4, mid_dim//4, 1 ,groups=mid_dim//4),
            nn.Conv2d(mid_dim//4, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=1)
        )
        

    def forward(self, inputs):
 
        
        x1, x2, x3, x4 = inputs # [16, 128, 128] [32, 64, 64] [64, 32, 32] [128, 16, 16]

        x1 = self.conv1(self.mlp1(x1))
        x2 = self.conv2(self.mlp2(x2))
        x3 = self.conv3(self.mlp3(x3))
        x4 = self.conv4(self.mlp4(x4))

        x2, x3, x4 = self.srg(x1, x2, x3, x4)  



        x1 = nn.Upsample(size=self.size, mode="bilinear")(x1)
        x2 = nn.Upsample(size=self.size, mode="bilinear")(x2)
        x3 = nn.Upsample(size=self.size, mode="bilinear")(x3)
        x4 = nn.Upsample(size=self.size, mode="bilinear")(x4)  #  [1, 16, 512, 512]

        x = self.block(torch.cat([x1,x2,x3,x4], dim=1))   # [1, 8, 512, 512]
        x = self.linear_pred(x)

        return x
