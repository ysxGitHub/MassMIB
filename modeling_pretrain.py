"""
@time: 2021/12/7

@ author: ysx
"""

import torch
import torch.nn as nn
from functools import partial
from net_base import PatchEmbed, Block, CrossBlock, EncoderDist, trunc_normal_, _init_weights, \
    get_sinusoid_encoding_table, compute_multi_view_skl1, compute_multi_view_skl2, get_pretrain_mse_loss, _cfg, \
    compute_multi_view_skl3, EncoderLatent, EncoderDiscrete, InceptionTime, VisionTransformer, CrossView, CnnStem
from timm.models.registry import register_model
from collections import OrderedDict


class MultiEncoder(nn.Module):
    def __init__(self,
                 in_c=12,
                 img_size=1000,
                 patch_size=10,
                 encoder_embed_dim=120,
                 encoder_depth=2,
                 encoder_num_heads=4,
                 mlp_ratio=4.0,
                 encoder_num_classes=0,
                 latent_dim=640,
                 drop_path_rate=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 is_img_view1=False,
                 is_img_view2=False,
                 use_mean_pooling=True,
                 is_using_low_dim=True,
                 ):
        super(MultiEncoder, self).__init__()
        # self.is_img_view1 = is_img_view1
        # self.is_img_view2 = is_img_view2
        self.oder_dim = 1280

        self.is_using_low_dim = is_using_low_dim

        self.view1_encoder = nn.Sequential(OrderedDict([
            ("cnn", InceptionTime(in_channel=in_c)),
            # ("cnn", CnnStem(in_channels=in_c)),
            ("vit", VisionTransformer(img_size=img_size, patch_size=patch_size, in_c=128, embed_dim=self.oder_dim,
                                      drop_path_rate=drop_path_rate, drop_rate=drop_rate, depth=2, norm_layer=norm_layer,
                                      mlp_ratio=mlp_ratio, num_heads=10, attn_drop_rate=attn_drop_rate,
                                      use_mean_pooling=use_mean_pooling, is_img=is_img_view1,
                                      use_learnable_pos_emb=True)),
        ]))

        self.view2_encoder = nn.Sequential(OrderedDict([
            ("vit", VisionTransformer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=encoder_embed_dim,
                                      drop_path_rate=drop_path_rate, drop_rate=drop_rate, depth=encoder_depth,
                                      norm_layer=norm_layer, mlp_ratio=mlp_ratio, num_heads=encoder_num_heads,
                                      attn_drop_rate=attn_drop_rate, use_mean_pooling=use_mean_pooling,
                                      is_img=is_img_view2, use_learnable_pos_emb=True)),
            ("fc", nn.Linear(encoder_embed_dim, self.oder_dim))
        ]))

        # self.cross_view = CrossAttention(dim=encoder_embed_dim, num_heads=encoder_num_heads)
        self.cross_view = CrossView(embed_dim=self.oder_dim, depth=1, num_heads=10, mlp_ratio=2.0,)

        if self.is_using_low_dim:
            self.encoder_dist1 = EncoderDist(self.oder_dim, self.oder_dim, self.oder_dim//2)
            self.encoder_dist2 = EncoderDist(self.oder_dim, self.oder_dim, self.oder_dim//2)
        else:
            self.encoder_dist1 = EncoderDiscrete(self.oder_dim, self.oder_dim//2)
            self.encoder_dist2 = EncoderDiscrete(self.oder_dim, self.oder_dim//2)
            # self.encoder_dist1 = EncoderLatent(encoder_embed_dim, latent_dim)
            # self.encoder_dist2 = EncoderLatent(encoder_embed_dim, latent_dim)

    def forward(self, x, y):
        # if not self.is_img_view1:
        #     x = x.unsqueeze(2)
        # if not self.is_img_view2:
        #     y = y.unsqueeze(2)
        output_x = self.view1_encoder(x)
        output_y = self.view2_encoder(y)
        output_x, output_y = self.cross_view(output_x, output_y)

        if self.is_using_low_dim:
            output_x, mean1, std1 = self.encoder_dist1(output_x)
            output_y, mean2, std2 = self.encoder_dist2(output_y)
            skl = compute_multi_view_skl1(mean1, std1, mean2, std2)
        else:
            z1_ = self.encoder_dist1(output_x)
            z2_ = self.encoder_dist2(output_y)
            skl = compute_multi_view_skl2(z1_, z2_)
            # output_x, mean1, std1 = self.encoder_dist1(output_x)
            # output_y, mean2, std2 = self.encoder_dist2(output_y)
            # skl = compute_multi_view_skl3(mean1, std1, mean2, std2)
        return output_x, output_y, skl


class Decoder(nn.Module):
    def __init__(self,
                 num_classes=120,
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
        super(Decoder, self).__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # self.proj = nn.ConvTranspose1d(embed_dim, 12, kernel_size=patch_size, stride=patch_size)
        # self.norm2 = norm_layer(1000)

    def forward(self, x):
        x = self.blocks(x)
        # x = self.proj(self.norm(x).transpose(1, 2))
        x = self.norm(x)
        return x


class MultiDecoder(nn.Module):
    def __init__(self,
                 patch_size=10,
                 latent_dim=60,
                 decoder_num_classes=120,
                 decoder_depth=2,
                 decoder_num_heads=4,
                 mlp_ratio=4.0,
                 normlize_target=True,
                 is_img_view1=False,
                 is_img_view2=False,
                 ):
        super(MultiDecoder, self).__init__()
        # self.normlize_target = normlize_target
        # self.is_img_view1 = is_img_view1
        # self.is_img_view2 = is_img_view2
        # self.patch_size = patch_size

        self.view1_decoder = Decoder(embed_dim=latent_dim,
                                     depth=decoder_depth,
                                     num_heads=decoder_num_heads,
                                     mlp_ratio=mlp_ratio)

        self.view2_decoder = Decoder(embed_dim=latent_dim,
                                     depth=decoder_depth,
                                     num_heads=decoder_num_heads,
                                     mlp_ratio=mlp_ratio)

        self.proj1 = nn.ConvTranspose1d(latent_dim, 12, kernel_size=patch_size, stride=patch_size)
        self.tan = nn.Tanh()
        self.proj2 = nn.ConvTranspose2d(latent_dim, 12, kernel_size=(5, 2), stride=(5, 2))
        # ???

    def forward(self, x, y):
        # if not self.is_img_view1:
        #     x = x.unsqueeze(2)
        # if not self.is_img_view2:
        #     y = y.unsqueeze(2)
        x1 = self.view1_decoder(x)
        x1 = self.tan(self.proj1(x1.transpose(1, 2)))

        x2 = self.view2_decoder(y)
        x2 = x2.transpose(1, 2)
        x2 = self.proj2(x2.reshape(x2.shape[0], x2.shape[1], 10, 10))

        return x1, x2


class PretrainMIBVisionTransformer(nn.Module):
    def __init__(self,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 in_c=12,
                 img_size=1000,
                 patch_size=10,
                 encoder_embed_dim=120,
                 encoder_depth=4,
                 encoder_num_heads=12,
                 encoder_mlp_ratio=4.0,
                 encoder_num_classes=0,
                 latent_dim=60,
                 decoder_num_classes=120,
                 decoder_depth=2,
                 decoder_num_heads=12,
                 decoder_mlp_ratio=4.0,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 is_using_low_dim=True,
                 normlize_target=True,
                 is_img_view1=False,
                 is_img_view2=False,
                 ):
        super(PretrainMIBVisionTransformer, self).__init__()
        self.mencoder = MultiEncoder(in_c=in_c, img_size=img_size, patch_size=patch_size,
                                     encoder_embed_dim=encoder_embed_dim, encoder_depth=encoder_depth,
                                     encoder_num_heads=encoder_num_heads, mlp_ratio=encoder_mlp_ratio,
                                     encoder_num_classes=encoder_num_classes, latent_dim=latent_dim,
                                     drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                                     is_img_view1=is_img_view1, is_img_view2=is_img_view2,
                                     is_using_low_dim=is_using_low_dim)
        # if not is_using_low_dim:
        #     latent_dim = latent_dim * 2
        # self.mencoder_to_decoder1 = nn.Linear(self.mencoder.oder_dim, latent_dim)
        # self.mencoder_to_decoder2 = nn.Linear(self.mencoder.oder_dim, latent_dim)

        self.mdecoder = MultiDecoder(patch_size=patch_size, latent_dim=latent_dim,
                                     decoder_num_classes=decoder_num_classes, decoder_depth=decoder_depth,
                                     decoder_num_heads=decoder_num_heads, mlp_ratio=decoder_mlp_ratio,
                                     normlize_target=normlize_target, is_img_view1=is_img_view1,
                                     is_img_view2=is_img_view2, )

        self.apply(_init_weights)

    def forward(self, x, y):
        z1_, z2_, skl = self.mencoder(x, y)

        # z1_ = self.mencoder_to_decoder1(z1_)
        # z2_ = self.mencoder_to_decoder2(z2_)

        x1, x2 = self.mdecoder(z1_, z2_)

        return x1, x2, skl


@register_model
def pretrain_mib_vit(pretrained=False, **kwargs):
    model = PretrainMIBVisionTransformer(
        in_c=12,
        img_size=1000,
        patch_size=10,
        encoder_embed_dim=120,
        encoder_depth=6,
        encoder_num_heads=12,
        encoder_num_classes=0,
        encoder_mlp_ratio=4.0,
        latent_dim=1280,
        decoder_num_classes=120,
        decoder_depth=2,
        decoder_num_heads=4,
        decoder_mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    # model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(' ')
        model.load_state_dict(checkpoint["model"])

    return model


if __name__ == '__main__':
    x = torch.rand(1, 12, 1000).cuda()
    y = torch.rand(1, 12, 50, 20).cuda()
    model = pretrain_mib_vit(is_img_view1=False, is_img_view2=True, is_using_low_dim=False).cuda()
    output = model(x, y)
    print(output[0].shape, output[1].shape)
    # print(model)
    # print(model.state_dict())


