"""
@time: 2021/12/6

@ author: ysx
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange
from functools import partial
from utils import device, MTS_DEFAULT_MEAN, MTS_DEFAULT_STD


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


def _init_weights(m):
    """
    weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=1000, patch_size=50, in_c=2, embed_dim=600, norm_layer=None, is_img=False):
        super().__init__()

        self.is_img = is_img

        if self.is_img:
            img_size = (50, 20)
            patch_size = (5, 2)
        else:
            img_size = (1, img_size)
            patch_size = (1, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # [64, 12, 1, 1000] => [64, 120, 1, 100] => [64, 120, 100]  => [64, 100, 120]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)

        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x, y):
        Bx, Nx, Cx = x.shape
        By, Ny, Cy = y.shape

        qkv_x = self.qkv_x(x).reshape(Bx, Nx, 3, self.num_heads, Cx // self.num_heads).permute(2, 0, 3, 1, 4)
        q_x, k_x, v_x = qkv_x[0], qkv_x[1], qkv_x[2]

        qkv_y = self.qkv_y(y).reshape(By, Ny, 3, self.num_heads, Cy // self.num_heads).permute(2, 0, 3, 1, 4)
        q_y, k_y, v_y = qkv_y[0], qkv_y[1], qkv_y[2]

        # q_x  k_y  v_y
        attn_xy = (q_x @ k_y.transpose(-2, -1)) * self.scale
        attn_xy = attn_xy.softmax(dim=-1)
        attn_xy = self.attn_drop(attn_xy)

        xy = (attn_xy @ v_y).transpose(1, 2).reshape(Bx, Nx, Cx)
        xy = self.proj_y(xy)
        xy = self.proj_drop(xy)

        # q_y  k_x  v_x
        attn_yx = (q_y @ k_x.transpose(-2, -1)) * self.scale
        attn_yx = attn_yx.softmax(dim=-1)
        attn_yx = self.attn_drop(attn_yx)

        yx = (attn_yx @ v_x).transpose(1, 2).reshape(By, Ny, Cy)
        yx = self.proj_x(yx)
        yx = self.proj_drop(yx)
        return yx, xy


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(CrossBlock, self).__init__()

        self.norm11 = norm_layer(dim)
        self.norm12 = norm_layer(dim)

        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.norm21 = norm_layer(dim)
        self.norm22 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x, y):
        x, y = self.attn(self.norm11(x), self.norm12(y))

        x = x + self.drop_path(x)
        y = y + self.drop_path(y)

        x = x + self.drop_path(self.mlp1(self.norm21(x)))
        y = y + self.drop_path(self.mlp2(self.norm22(y)))

        return x, y


class InceptionBaseBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(InceptionBaseBlock, self).__init__()
        self.out_planes = out_planes//4

        self.bottleneck = nn.Conv1d(in_planes, self.out_planes, kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv1d(self.out_planes, self.out_planes, kernel_size=39, stride=1, padding=19, bias=False)
        self.conv3 = nn.Conv1d(self.out_planes, self.out_planes, kernel_size=19, stride=1, padding=9, bias=False)
        self.conv2 = nn.Conv1d(self.out_planes, self.out_planes, kernel_size=9, stride=1, padding=4, bias=False)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.conv1 = nn.Conv1d(in_planes, self.out_planes, kernel_size=1, stride=1, bias=False)

        self.bn = nn.BatchNorm1d(self.out_planes * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.bottleneck(x)
        output4 = self.conv4(output)
        output3 = self.conv3(output)
        output2 = self.conv2(output)

        output1 = self.maxpool(x)
        output1 = self.conv1(output1)

        x_out = self.relu(self.bn(torch.cat((output1, output2, output3, output4), dim=1)))
        return x_out


class InceptionTime(nn.Module):
    def __init__(self, in_channel=12):
        super(InceptionTime, self).__init__()

        self.BaseBlock1 = InceptionBaseBlock(in_channel, 128)
        self.BaseBlock2 = InceptionBaseBlock(128, 128)
        self.BaseBlock3 = InceptionBaseBlock(128, 128)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channel, 128, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(128)

        # self.BaseBlock4 = InceptionBaseBlock(128)
        # self.BaseBlock5 = InceptionBaseBlock(128)
        # self.BaseBlock6 = InceptionBaseBlock(128)

        # self.conv2 = nn.Conv1d(128, 128, kernel_size=1, stride=1, bias=False)
        # self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        shortcut1 = self.bn1(self.conv1(x))
        output1 = self.BaseBlock1(x)
        output1 = self.BaseBlock2(output1)
        output1 = self.BaseBlock3(output1)
        output1 = self.relu(output1 + shortcut1)
        # output1 = self.relu(self.bn2(output1 + shortcut1))

        # shortcut2 = self.bn2(self.conv2(output1))
        # output2 = self.BaseBlock4(output1)
        # output2 = self.BaseBlock5(output2)
        # output2 = self.BaseBlock6(output2)
        # output2 = self.relu(output2 + shortcut2)

        return output1


class CnnStem(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(CnnStem, self).__init__()
        self.layer1 = nn.Sequential(*[
            nn.Conv1d(in_channels, 256, kernel_size=25, dilation=1, stride=1, padding=12),
            nn.BatchNorm1d(256)
            ])
        self.layer2 = nn.Sequential(*[
            nn.Conv1d(256, 256, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(256)
            ])
        self.layer3 = nn.Sequential(*[
            nn.Conv1d(256, 128, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(128)
        ])
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=1000,
                 patch_size=10,
                 in_c=12,
                 num_classes=0,
                 embed_dim=120,
                 depth=2,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=None,
                 use_learnable_pos_emb=False,
                 # init_scale=0.,
                 use_mean_pooling=True,
                 is_img=False):
        super().__init__()
        self.is_img = is_img
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim,
                                      is_img=is_img)
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_rate, attn_drop_ratio=attn_drop_rate, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        # self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.head.weight, std=.02)
        # self.apply(_init_weights)
        # self.head.weight.data.mul_(init_scale)
        # self.head.bias.data.mul_(init_scale)

    def forward_features(self, x):
        if not self.is_img:
            x = x.unsqueeze(2)

        x = self.patch_embed(x)
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)

        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
        # if self.fc_norm is not None:
        #     # return self.fc_norm(x[:, 1:].mean(1))
        #     return self.fc_norm(x.mean(1))
        # else:
        #     return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class CrossView(nn.Module):
    def __init__(self,
                 embed_dim=120,
                 depth=2,
                 num_heads=4,
                 mlp_ratio=2.0,
                 norm_layer=None,
                 qkv_bias=True,
                 qk_scale=None,
                 act_layer=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.):
        super(CrossView, self).__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            CrossBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                       norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

    def forward(self, x, y):
        for blk in self.blocks:
            x, y = blk(x, y)

        x = self.norm1(x)
        y = self.norm2(y)
        return x, y



class EncoderDist(nn.Module):
    def __init__(self, embed_size, hidden_size, latent_dim, activation=nn.ELU, distribution=td.Normal):
        super(EncoderDist, self).__init__()

        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.act = activation()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_dim)

        self._dist = distribution
        self._ln_stoch = nn.LayerNorm(latent_dim // 2)

    def forward(self, embed):
        outputs = self.fc2(self.ln(self.act(self.fc1(embed))))
        mean, std = torch.chunk(outputs, 2, dim=2)
        std = F.softplus(std) + 1e-7
        dist = self._dist(mean, std)
        enc_dist = dist.rsample()
        enc_dist = self._ln_stoch(enc_dist)
        return enc_dist, mean, std


def get_dist(mean, std):
    return td.independent.Independent(td.Normal(mean, std), 1)


def compute_multi_view_skl1(mean1, std1, mean2, std2):
    z1_dist = get_dist(mean1, std1)
    z2_dist = get_dist(mean2, std2)
    kl_1_2 = torch.mean(
        torch.distributions.kl.kl_divergence(z1_dist, z2_dist)
    )
    kl_2_1 = torch.mean(
        torch.distributions.kl.kl_divergence(z2_dist, z1_dist)
    )
    skl = (kl_1_2 + kl_2_1) / 2.
    return skl


class EncoderDiscrete(nn.Module):
    def __init__(self, embed_size, latent_dim, activation=nn.ELU, distribution=td.Normal):
        super(EncoderDiscrete, self).__init__()

        self._dist = distribution
        self.act = activation()

        self.fc11 = nn.Linear(embed_size, latent_dim)
        self.ln1 = nn.LayerNorm(latent_dim)

        self.fc12 = nn.Linear(embed_size, latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)

    def forward(self, embed):
        mean = self.ln1(self.act(self.fc11(embed)))
        std = self.ln2(self.act(self.fc12(embed)))
        std = F.softplus(std) + 1e-7
        dist = self._dist(mean, std)

        return dist


def compute_multi_view_skl2(input1, input2):

    z1 = input1.rsample()
    z2 = input2.rsample()

    # Symmetrized Kullback-Leibler divergence
    kl_1_2 = input1.log_prob(z1) - input2.log_prob(z1)
    kl_2_1 = input2.log_prob(z2) - input1.log_prob(z2)
    skl = (kl_1_2 + kl_2_1).mean() / 2.
    return skl


class EncoderLatent(nn.Module):
    def __init__(self, embed_size, latent_dim):
        super(EncoderLatent, self).__init__()
        self.encoder_mean = nn.Linear(embed_size, latent_dim)
        self.encoder_std = nn.Linear(embed_size, latent_dim)

    def forward(self, embed):
        encoder_mean = self.encoder_mean(embed)
        encoder_std = F.softplus(self.encoder_std(embed))

        # sample latent based on encoder outputs
        latent_dist = td.Normal(encoder_mean, encoder_std)
        latent = latent_dist.sample()
        return latent, encoder_mean, encoder_std


def compute_multi_view_skl3(encoder_mean1, encoder_std1, encoder_mean2, encoder_std2):
    prior1 = td.independent.Independent(td.Normal(
        torch.zeros_like(encoder_mean1), torch.ones_like(encoder_mean1)), 1)
    prior2 = td.independent.Independent(td.Normal(
        torch.zeros_like(encoder_mean2), torch.ones_like(encoder_mean2)), 1)

    enc_dist1 = get_dist(encoder_mean1, encoder_std1)
    enc_dist2 = get_dist(encoder_mean2, encoder_std2)

    kl_vec1 = torch.distributions.kl.kl_divergence(enc_dist1, prior1)
    kl_vec2 = torch.distributions.kl.kl_divergence(enc_dist2, prior2)

    kl_loss = (kl_vec1.mean() + kl_vec2.mean())/2.

    return kl_loss


def get_pretrain_mse_loss(inputs, outputs,
                          bool_masked_pos,
                          loss_func,
                          patch_size=10,
                          normlize_target=True,
                          is_img=False):
    with torch.no_grad():

        if is_img:
            patch_size1 = 5
            patch_size2 = 2
        else:
            patch_size1 = 1
            patch_size2 = patch_size

        # calculate the predict label
        mean = torch.as_tensor(MTS_DEFAULT_MEAN).to(device)[None, :, None, None]
        std = torch.as_tensor(MTS_DEFAULT_STD).to(device)[None, :, None, None]
        unnorm_inputs = inputs * std + mean  # in [0, 1]

        if normlize_target:
            images_squeeze = rearrange(unnorm_inputs, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size1,
                                       p2=patch_size2)
            images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                           ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            # we find that the mean is about 0.48 and standard deviation is about 0.08.
            images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
        else:
            images_patch = rearrange(unnorm_inputs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size1,
                                     p2=patch_size2)

        B, _, C = images_patch.shape
        labels = images_patch[bool_masked_pos].reshape(B, -1, C)

    mse_loss = loss_func(input=outputs, target=labels)

    return labels, mse_loss


if __name__ == '__main__':
    model = CnnStem(in_channels=12,).cuda()
    x = torch.randn(64, 12, 1000).cuda()
    print(model(x).shape)


