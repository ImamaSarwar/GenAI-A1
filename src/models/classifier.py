"""4-class corruption classifier: [clean, salt_pepper, blur, occlusion]. Returns raw logits."""
import torch.nn as nn


class CorruptionCNN(nn.Module):
    def __init__(self, base_ch=32, dropout=0.3, n_classes=4):
        super().__init__()
        c = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        layers, cin = [], 3
        for co in c:
            layers += [nn.Conv2d(cin, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                       nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                       nn.MaxPool2d(2)]
            cin = co
        self.features = nn.Sequential(*layers)          # 128 -> 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c[-1], n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))
