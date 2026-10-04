"""
Denoising autoencoder with a genuine bottleneck (NO skip connections).

Encoder : 5 x [Conv 4x4 stride 2 -> BN -> LeakyReLU]   128 -> 64 -> 32 -> 16 -> 8 -> 4
          channels: C, 2C, 4C, 8C, 8C
Latent  : flatten (8C*4*4) -> Linear -> latent_dim -> Dropout
Decoder : Linear -> reshape (8C,4,4) -> 5 x [Upsample x2 -> Conv3x3 -> BN -> ReLU] -> Conv -> Sigmoid
Upsample+Conv is used instead of ConvTranspose to avoid checkerboard artifacts (Odena et al., 2016).
"""
import torch
import torch.nn as nn


class DAE(nn.Module):
    def __init__(self, base_ch=32, latent_dim=256, dropout=0.1):
        super().__init__()
        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]
        self.c_last = c[-1]
        self.flat = c[-1] * 4 * 4

        enc, cin = [], 3
        for co in c:
            enc += [nn.Conv2d(cin, co, 4, 2, 1, bias=False), nn.BatchNorm2d(co), nn.LeakyReLU(0.2, True)]
            cin = co
        self.encoder = nn.Sequential(*enc)
        self.to_latent = nn.Linear(self.flat, latent_dim)
        self.drop = nn.Dropout(dropout)

        self.from_latent = nn.Sequential(nn.Linear(latent_dim, self.flat), nn.LeakyReLU(0.2, True))
        dec, cin = [], c[-1]
        for co in [c[3], c[2], c[1], c[0], c[0]]:
            dec += [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(cin, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True)]
            cin = co
        dec += [nn.Conv2d(cin, 3, 3, 1, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.drop(self.to_latent(self.encoder(x).flatten(1)))

    def decode(self, z):
        return self.decoder(self.from_latent(z).view(-1, self.c_last, 4, 4))

    def forward(self, x):
        return self.decode(self.encode(x))
