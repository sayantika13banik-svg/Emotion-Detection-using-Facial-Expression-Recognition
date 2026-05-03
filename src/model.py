import torch
import torch.nn as nn
# torch.nn — contains all neural network building blocks

class EmotionCNN(nn.Module):
    # nn.Module — base class for all PyTorch models
    # every custom model must inherit from it

    def __init__(self):
        super(EmotionCNN, self).__init__()
        # super().__init__() — required, initialises nn.Module

        # ── BLOCK 1 ──────────────────────────────────────────
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # in_channels=1  — grayscale image, 1 channel
        # out_channels=32 — learn 32 different filters
        # kernel_size=3   — each filter is 3×3
        # padding=1       — keeps output same size as input

        self.bn1 = nn.BatchNorm2d(32)
        # normalizes output of conv1
        # keeps values stable during training

        # ── BLOCK 2 ──────────────────────────────────────────
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # ── BLOCK 3 ──────────────────────────────────────────
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # ── SHARED LAYERS ─────────────────────────────────────
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        # shrinks image by half each time
        # 48×48 → 24×24 → 12×12 → 6×6

        self.relu    = nn.ReLU()
        # replaces negative values with 0
        # adds non-linearity so model learns complex patterns

        self.dropout = nn.Dropout(p=0.5)
        # randomly turns off 50% neurons during training
        # prevents overfitting

        # ── FULLY CONNECTED ───────────────────────────────────
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        # 128 filters × 6×6 image size after 3 pooling layers = 4608
        # 256 = neurons in this layer

        self.fc2 = nn.Linear(256, 7)
        # 7 = number of emotion classes
        # final output — one score per emotion

    def forward(self, x):
        # defines how data flows through layers
        # x = input tensor (batch, 1, 48, 48)

        # ── BLOCK 1 ──────────────────────────────────────────
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # conv1 → bn1 → relu → pool
        # (batch,1,48,48) → (batch,32,24,24)

        # ── BLOCK 2 ──────────────────────────────────────────
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # (batch,32,24,24) → (batch,64,12,12)

        # ── BLOCK 3 ──────────────────────────────────────────
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # (batch,64,12,12) → (batch,128,6,6)

        # ── FLATTEN ───────────────────────────────────────────
        x = x.view(x.size(0), -1)
        # (batch,128,6,6) → (batch,4608)
        # Linear layers need 1D input

        # ── FULLY CONNECTED ───────────────────────────────────
        x = self.relu(self.fc1(x))
        # (batch,4608) → (batch,256)

        x = self.dropout(x)
        # randomly drop 50% neurons — only active during training

        x = self.fc2(x)
        # (batch,256) → (batch,7)
        # 7 raw scores — one per emotion

        return x

if __name__ == "__main__":
    # this block only runs when you run model.py directly
    # not when it's imported by webcam.py
    # useful for testing the model in isolation

    model = EmotionCNN()
    test  = torch.randn(4, 1, 48, 48)
    # 4 fake images, 1 channel, 48×48

    out   = model(test)
    print("Output shape:", out.shape)
    # should print: torch.Size([4, 7])