import torch
import torch.nn as nn

class DAE(nn.Module):
    """Vector-bottleneck Denoising Autoencoder."""
    def __init__(self, base_ch=32, latent_dim=256, dropout=0.1):
        super().__init__()
        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        enc, cin = [], 3
        for co in c:
            enc += [nn.Conv2d(cin, co, 4, 2, 1, bias=False), nn.BatchNorm2d(co), nn.LeakyReLU(0.2, True)]
            cin = co
        self.encoder = nn.Sequential(*enc)
        self.fc_enc = nn.Linear(c[-1] * 8 * 8, latent_dim)
        self.drop = nn.Dropout(dropout)

        self.fc_dec = nn.Linear(latent_dim, c[-1] * 8 * 8)
        dec = []
        cin = c[-1]
        for co in [c[2], c[1], c[0], c[0]]:
            dec += [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(cin, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True)]
            cin = co
        dec += [nn.Conv2d(cin, 3, 3, 1, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.drop(self.fc_enc(h))

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), -1, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvDAE(nn.Module):
    """
    Denoising autoencoder with a SPATIAL bottleneck (still no skip connections).
    Encoder: 4 x [Conv 4x4 s2 -> BN -> LeakyReLU]  128 -> 8x8, channels C,2C,4C,8C
             then 1x1 conv -> latent_ch  => latent is (latent_ch, 8, 8), i.e. 64*latent_ch values
    Decoder: 3x3 conv -> 4 x [Upsample x2 -> Conv3x3 -> BN -> ReLU] -> Conv -> Sigmoid
    """
    def __init__(self, base_ch=32, latent_ch=16, dropout=0.1):
        super().__init__()
        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        enc, cin = [], 3
        for co in c:
            enc += [nn.Conv2d(cin, co, 4, 2, 1, bias=False), nn.BatchNorm2d(co), nn.LeakyReLU(0.2, True)]
            cin = co
        self.encoder = nn.Sequential(*enc)
        self.to_latent = nn.Conv2d(c[-1], latent_ch, 1)
        self.drop = nn.Dropout(dropout)
        self.latent_size = latent_ch * 8 * 8

        dec = [nn.Conv2d(latent_ch, c[-1], 3, 1, 1, bias=False), nn.BatchNorm2d(c[-1]), nn.ReLU(True)]
        cin = c[-1]
        for co in [c[2], c[1], c[0], c[0]]:
            dec += [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(cin, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True)]
            cin = co
        dec += [nn.Conv2d(cin, 3, 3, 1, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.drop(self.to_latent(self.encoder(x)))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


def build_model(cfg):
    """arch='conv' -> spatial bottleneck (default); arch='vec' -> flat vector bottleneck."""
    if cfg.get("arch", "conv") == "vec":
        return DAE(cfg.get("base_ch", 32), cfg.get("latent_dim", 256), cfg.get("dropout", 0.1))
    return ConvDAE(cfg.get("base_ch", 32), cfg.get("latent_ch", 16), cfg.get("dropout", 0.1))
