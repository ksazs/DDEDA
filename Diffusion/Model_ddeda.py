import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch.fft as fft


# Swish Activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False).to('cuda:0')
        self.softmax = nn.Softmax(2).to('cuda:0')
        # Choose to deploy A0 on GPU or CPU according to your needs
        self.A0 = torch.eye(hide_channel).to('cuda:0')
        # self.A0 = torch.eye(hide_channel)
        # A2 is initialized to 1e-6
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))).to('cuda:0'), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False).to('cuda:0')
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False).to('cuda:0')
        self.relu = nn.ReLU(inplace=True).to('cuda:0')
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False).to('cuda:0')
        self.sigmoid = nn.Sigmoid().to('cuda:0')

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1) + self.A2

        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y

# Fourier Filter
def Fourier_filter(x, threshold, scale):
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered


# Time Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        super().__init__()
        assert d_model % 2 == 0
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1).view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        return self.timembedding(t)


# DownSample Layer
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb=None):
        return self.main(x)


# UpSample Layer
class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.main(x)


# Attention Block
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h).permute(0, 2, 3, 1).view(B, H * W, C)
        k = self.proj_k(h).view(B, C, H * W)
        w = F.softmax(torch.bmm(q, k) * (C ** -0.5), dim=-1)

        v = self.proj_v(h).permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v).view(B, H, W, C).permute(0, 3, 1, 2)
        return x + self.proj(h)


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, ratio=4):
        super().__init__()
        self.block1 = nn.Sequential(nn.GroupNorm(32, in_ch), Swish(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.temb_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))
        self.block2 = nn.Sequential(nn.GroupNorm(32, out_ch), Swish(), nn.Dropout(dropout),
                                    nn.Conv2d(out_ch, out_ch, 3, padding=1))
        # 添加AGCA模块
        self.agca = AGCA(out_ch, ratio)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        # 在block2之后调用AGCA
        h = self.agca(h)
        return self.attn(h + self.shortcut(x))


# Free_UNetModel
class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, b1=1.2, b2=1.4, s1=0.9, s2=0.2):
        super().__init__()
        self.time_embedding = TimeEmbedding(T, ch, ch * 4)
        self.head = nn.Conv2d(3, ch, 3, padding=1)

        self.downblocks = nn.ModuleList()
        self.middleblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()

        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(now_ch, out_ch, ch * 4, dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([ResBlock(now_ch, now_ch, ch * 4, dropout, attn=True)])

        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(chs.pop() + now_ch, out_ch, ch * 4, dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        self.tail = nn.Sequential(nn.GroupNorm(32, now_ch), Swish(), nn.Conv2d(now_ch, 3, 3, padding=1))
        self.b1, self.b2, self.s1, self.s2 = b1, b2, s1, s2

    def forward(self, x, t):
        temb = self.time_embedding(t)
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        for layer in self.middleblocks:
            h = layer(h, temb)

        for i, layer in enumerate(self.upblocks):
            if isinstance(layer, ResBlock):
                hs_ = hs.pop()
                if h.shape[1] == 1280:
                    h[:, :640] *= (self.b1 - 1) * h.mean(1, keepdim=True) + 1
                    hs_ = Fourier_filter(hs_, 1, self.s1)
                elif h.shape[1] == 640:
                    h[:, :320] *= (self.b2 - 1) * h.mean(1, keepdim=True) + 1
                    hs_ = Fourier_filter(hs_, 1, self.s2)
                h = torch.cat([h, hs_], dim=1)
            h = layer(h, temb)

        return self.tail(h)


# Main
if __name__ == "__main__":
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        T=1000, ch=64, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1, b1=1.2, b2=1.4, s1=0.9, s2=0.2
    ).to(device)
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    t = torch.randint(1000, (batch_size,)).to(device)
    y = model(x, t)
    print(y.shape)