"""
@time: 2021/12/8

@ author: ysx
"""
import torch
import torch.nn as nn
from functools import partial
from timm.models.registry import register_model
from net_base import PatchEmbed, Block, CrossBlock, EncoderDist, trunc_normal_, _init_weights, \
    get_sinusoid_encoding_table, compute_multi_view_skl1, compute_multi_view_skl2, _cfg, EncoderLatent, \
    compute_multi_view_skl3, EncoderDiscrete, InceptionTime, VisionTransformer, CrossView, CnnStem
import torch.distributions as td
from collections import OrderedDict


class MIBVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,  # avoid the error from create_fn in timm
                 num_classes=10,
                 img_size=1000,
                 patch_size=10,
                 in_c=12,
                 embed_dim=120,
                 depth=2,
                 num_heads=12,
                 mlp_ratio=4.0,
                 latent_dim=640,
                 init_scale=0,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 use_mean_pooling=True,
                 is_img_view1=True,
                 is_img_view2=True,
                 is_using_low_dim=True,
                 ):
        super(MIBVisionTransformer, self).__init__()
        oder_dim = 1280
        # self.is_img_view1 = is_img_view1
        # self.is_img_view2 = is_img_view2

        self.is_using_low_dim = is_using_low_dim

        self.view1_encoder = nn.Sequential(OrderedDict([
            ("cnn", InceptionTime(in_channel=in_c)),
            # ("cnn", CnnStem(in_channels=in_c)),
            ("vit", VisionTransformer(img_size=img_size, patch_size=patch_size, in_c=128, embed_dim=oder_dim,
                                      drop_path_rate=drop_path_rate, drop_rate=drop_rate, depth=2,
                                      mlp_ratio=mlp_ratio, num_heads=10, use_mean_pooling=use_mean_pooling,
                                      attn_drop_rate=attn_drop_rate, is_img=is_img_view1,
                                      use_learnable_pos_emb=True)),
        ]))

        self.view2_encoder = nn.Sequential(OrderedDict([
            ("vit", VisionTransformer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim,
                                      drop_path_rate=drop_path_rate, drop_rate=drop_rate, depth=depth,
                                      mlp_ratio=mlp_ratio, num_heads=num_heads,use_mean_pooling=use_mean_pooling,
                                      attn_drop_rate=attn_drop_rate, is_img=is_img_view2,
                                      use_learnable_pos_emb=True)),
            ("fc", nn.Linear(embed_dim, oder_dim))
        ]))

        # self.cross_view = CrossAttention(dim=embed_dim, num_heads=num_heads)
        self.cross_view = CrossView(embed_dim=oder_dim, depth=1, num_heads=10, mlp_ratio=2.0, )

        if self.is_using_low_dim:
            self.encoder_dist1 = EncoderDist(oder_dim, latent_dim * 2, latent_dim)
            self.encoder_dist2 = EncoderDist(oder_dim, latent_dim * 2, latent_dim)
        else:
            self.encoder_dist1 = EncoderDiscrete(oder_dim, latent_dim)
            self.encoder_dist2 = EncoderDiscrete(oder_dim, latent_dim)
            # self.encoder_dist1 = EncoderLatent(embed_dim, latent_dim)
            # self.encoder_dist2 = EncoderLatent(embed_dim, latent_dim)

        if not is_using_low_dim:
            latent_dim = latent_dim * 2
        else:
            embed_dim = embed_dim // 2

        self.fc_norm1 = norm_layer(oder_dim) if use_mean_pooling else None
        self.fc_norm2 = norm_layer(oder_dim) if use_mean_pooling else None

        self.head = nn.Linear(oder_dim * 2, num_classes) if num_classes > 0 else nn.Identity()

        # self.fuse_weight_1 = AdaptiveWeight(embed_dim)
        # self.fuse_weight_2 = AdaptiveWeight(embed_dim)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(_init_weights)
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        # blocks + cross_view + encoder_dist
        return len(self.view2_encoder.vit.blocks) + len(self.cross_view.blocks) + 1

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

        if self.fc_norm1 is not None:
            z1 = self.fc_norm1(output_x.mean(1))
            z2 = self.fc_norm2(output_y.mean(1))
        else:
            z1 = output_x[:, 0]
            z2 = output_y[:, 0]

        x = self.head(torch.cat([z1, z2], dim=-1))
        return x, skl


@register_model
def mib_vit_net(pretrained=False, **kwargs):
    model = MIBVisionTransformer(
        img_size=1000,
        patch_size=10,
        in_c=12,
        embed_dim=120,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        latent_dim=640,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    # model.default_cfg = _cfg()
    return model


if __name__ == '__main__':

    x = torch.rand(64, 12, 1000).cuda()
    y = torch.rand(64, 12, 50, 20).cuda()

    # x = torch.rand(64, 3, 224, 224).cuda()
    # y = torch.rand(64, 3, 224, 224).cuda()

    model = mib_vit_net(is_img_view1=False, is_img_view2=True, is_using_low_dim=False).cuda()
    print(model(x, y)[0].shape)
    # for name, param in model.named_parameters():
    #     print(name, param)