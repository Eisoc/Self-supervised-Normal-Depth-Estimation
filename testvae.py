import pytorch_lightning as pl
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from submodules.vae.vae_model import Decoder, Encoder
from submodules.vae.distributions import \
    DiagonalGaussianDistribution
from utils.util_vae import filter_nan_loss, instantiate_from_config
# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

class AutoencoderKL(pl.LightningModule):

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key='image',
                 colorize_nlabels=None,
                 monitor=None,
                 prior_model=None,
                 prior_normal=None,
                 using_rgb=True):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.prior_model = prior_model
        self.using_rgb = using_rgb

        assert ddconfig['double_z']
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig['z_channels'],
                                          2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim,
                                               ddconfig['z_channels'], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer('colorize',
                                 torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if prior_model is not None:
            self.prior_model = instantiate_from_config(prior_model)
        if prior_normal is not None:
            self.prior_normal = instantiate_from_config(prior_normal)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        try:
            sd = torch.load(path, map_location='cpu')['state_dict']
        except:
            sd = torch.load(path, map_location='cpu')

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Deleting key {} from state_dict.'.format(k))
                    del sd[k]
        m, u = self.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print('missing keys:')
            print(m)
        if len(u) > 0:
            print('unexpected keys:')
            print(u)

        print(f'Restored from {path}')

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def prior_to_eval(self):

        if self.prior_model is not None:
            self.prior_model.eval()

        if self.prior_normal is not None:
            self.prior_normal.eval()

    @torch.no_grad()
    def prior_inference(self, inputs, prior_inputs):
        # depth prior model
        # midas or zoe is 384 model
        prior_results = {}

        self.prior_to_eval()

        model_prior_results = self.prior_model(prior_inputs)
        prior_results.update(model_prior_results)

        # using normal map
        if not self.using_rgb:
            normal_prior = self.prior_normal(prior_inputs)
            prior_results.update(normal_prior)

        resize_prior_results = {}
        _, __, h, w = inputs.shape

        for key in prior_results.keys():
            resize_prior_results[key] = F.interpolate(
                prior_results[key], (w, h), mode='bilinear')

        if self.using_rgb:
            return torch.cat([inputs, resize_prior_results['depth']], dim=1)
        else:
            return torch.cat([
                resize_prior_results['normal'], resize_prior_results['depth']
            ],
                             dim=1)

    def forward(self, input, sample_posterior=True):

        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    
ddconfig = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 4,
    "out_ch": 4,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0
}

lossconfig = {
    "target": "torch.nn.Identity"
}

# 初始化 AutoencoderKL 类
autoencoder = AutoencoderKL(
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    embed_dim=4,
    monitor="val/rec_loss"
).cuda()

if __name__ == '__main__':
    x = torch.randn(4,4,128,416)
    posterior = autoencoder.encode(x)
    print(posterior.mean.shape)
    print(posterior.var.shape)