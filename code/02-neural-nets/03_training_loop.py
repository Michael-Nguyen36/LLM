import torch
import torch.nn as nn

# 03_training_loop.py — The full training loop: forward → loss → backward → step.
# This is the core recipe behind every neural network, from tiny MLPs to GPT-4.

# --- 1. Create synthetic classification data ---
# 200 data points, 10 features each. 3 classes (0, 1, 2).
# We embed a REAL pattern so the model can actually learn:
#   Class 0: first feature is negative
#   Class 1: first feature is positive, second is negative
#   Class 2: both are positive
num_samples, num_features, num_classes = 200, 10, 3
torch.manual_seed(42)                                # reproducible results
X = torch.randn(num_samples, num_features) * 0.5
y = torch.zeros(num_samples, dtype=torch.long)
y[(X[:, 0] > 0.2) & (X[:, 1] < 0.0)] = 1
y[(X[:, 0] > 0.2) & (X[:, 1] >= 0.0)] = 2
print(f"Data shape:   {X.shape}")
print(f"Class counts: 0={(y==0).sum().item()}, 1={(y==1).sum().item()}, 2={(y==2).sum().item()}")
print(f"Random guess accuracy: ~{100.0/num_classes:.0f}%")
print()

# --- 2. Define a simple classifier (10 → 16 → 8 → 3) ---
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)     # 3 output scores (one per class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))     # 10 → 16
        x = self.relu(self.fc2(x))     # 16 → 8
        x = self.fc3(x)                # 8 → 3  (raw scores, no activation)
        return x

model = Classifier()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# --- 3. Loss function and optimizer ---
loss_fn = nn.CrossEntropyLoss()                      # measures prediction error
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# --- 4. Training loop ---
# Split data: 80% train, 20% test — never evaluate on training data in practice!
n_train = 160
X_train, y_train = X[:n_train], y[:n_train]
X_test,  y_test  = X[n_train:], y[n_train:]

epochs = 100
print(f"\nTraining for {epochs} epochs...\n")

for epoch in range(epochs):
    # Forward pass: run data through the model → get predictions
    logits = model(X_train)                            # (160, 10) → (160, 3)

    # Compute loss: how wrong are our predictions?
    loss = loss_fn(logits, y_train)                    # single scalar

    # Zero old gradients, run backward, update weights
    optimizer.zero_grad()                              # clear gradients from previous step
    loss.backward()                                    # compute NEW gradients
    optimizer.step()                                   # nudge weights opposite to gradients

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1:3d}  |  Loss: {loss.item():.4f}")

print(f"\nFinal training loss: {loss.item():.4f}")
print()

# --- 5. Evaluate on TEST data (not training data — that would overstate accuracy) ---
with torch.no_grad():                                  # no gradients needed for evaluation
    logits = model(X_test)                             # (40, 10) → (40, 3)
    preds = torch.argmax(logits, dim=1)                # pick highest-score class
    acc = (preds == y_test).float().mean().item()

print(f"Accuracy on test data: {acc * 100:.1f}%  "
      f"(random chance = {100.0 / num_classes:.1f}%)")
print()

# --- Big-picture takeaway ---
print("This same pattern — forward, loss, backward, step — is how every")
print("LLM is trained, just with billions of parameters instead of hundreds.")
