import torch
import torch.nn as nn

# 02_mlp.py — Build a tiny Multi-Layer Perceptron (MLP) with nn.Module.
# Same pattern used in LLMs — just WAY bigger, with attention layers added.

# --- Define a tiny MLP with one hidden layer (16 → 8 → 4) ---
class TinyMLP(nn.Module):
    """A neural network with one hidden layer of 8 neurons."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)          # 16 inputs → 8 hidden neurons
        self.fc2 = nn.Linear(8, 4)           # 8 hidden → 4 outputs
        self.relu = nn.ReLU()                # ReLU sets negatives to 0 (adds non-linearity)

    def forward(self, x):
        x = self.fc1(x)                      # (batch, 16) → (batch, 8)
        print(f"    After fc1:  {x.shape}")
        x = self.relu(x)                     # shape stays (batch, 8)
        print(f"    After relu: {x.shape}")
        x = self.fc2(x)                      # (batch, 8) → (batch, 4)
        print(f"    After fc2:  {x.shape}")
        return x

print("=== TinyMLP (16 → 8 → 4) ===")
model = TinyMLP()
out = model(torch.randn(4, 16))              # 4 samples, 16 features each
print(f"Final output shape: {out.shape}")

# Count all trainable parameters
n = sum(p.numel() for p in model.parameters())
print(f"TinyMLP has {n} parameters total")
print()

# --- A deeper MLP with two hidden layers (16 → 32 → 16 → 4) ---
class TinyMLP2(nn.Module):
    """Two hidden layers."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))           # 16 → 32
        print(f"    After fc1+relu: {x.shape}")
        x = self.relu(self.fc2(x))           # 32 → 16
        print(f"    After fc2+relu: {x.shape}")
        x = self.fc3(x)                      # 16 → 4
        print(f"    After fc3:      {x.shape}")
        return x

print("=== TinyMLP2 (16 → 32 → 16 → 4) ===")
model2 = TinyMLP2()
out2 = model2(torch.randn(4, 16))
print(f"Final output shape: {out2.shape}")
n2 = sum(p.numel() for p in model2.parameters())
print(f"TinyMLP2 has {n2} parameters")

print()
print("This is exactly how LLMs are structured — just WAY bigger")
print("and with attention layers between the linear layers.")
