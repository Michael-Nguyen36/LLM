import torch
# (nn and F not needed here — they appear in Chapter 02 for building neural networks)

# 01_tensor_basics.py — What are tensors? Scalars, vectors, matrices, 3D tensors.
# A tensor = multi-dimensional array of numbers (the fundamental data in PyTorch).

# --- 0D Tensor: a scalar (single number) ---
scalar = torch.tensor(42)
print(f"Scalar: {scalar}")
print(f"  Shape: {scalar.shape}")           # torch.Size([]) — 0 dims
print(f"  ndim:  {scalar.ndim}")
print()

# --- 1D Tensor: a vector (list of numbers) ---
vector = torch.tensor([1, 2, 3, 4])
print(f"Vector: {vector}")
print(f"  Shape: {vector.shape}")           # torch.Size([4]) — 1 dim
print(f"  ndim:  {vector.ndim}")
print()

# --- 2D Tensor: a matrix (rows x columns) ---
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
print(f"Matrix:\n{matrix}")
print(f"  Shape: {matrix.shape}")           # torch.Size([3, 3]) — 2 dims
print()

# --- 3D Tensor: a stack of matrices ---
tensor_3d = torch.tensor([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]]])
print(f"3D Tensor:\n{tensor_3d}")
print(f"  Shape: {tensor_3d.shape}")        # torch.Size([2, 2, 2]) — 3 dims
print()

# --- Creating tensors with special fill patterns ---
zeros = torch.zeros((2, 3))                 # every element = 0.0
print(f"Zeros:\n{zeros}")

ones = torch.ones((2, 3))                   # every element = 1.0
print(f"Ones:\n{ones}")

random_vals = torch.randn((2, 3))           # random (normal distribution)
print(f"Random normal:\n{random_vals}")
print()

# --- Tensor properties: dtype and device ---
t = torch.tensor([1.5, 2.5, 3.5])
print(f"dtype:  {t.dtype}")     # torch.float32 (default for floats)
print(f"device: {t.device}")    # cpu (or cuda if GPU available)
print()

# --- Arithmetic: element-wise (*) vs. matrix multiply (@) ---
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

elem = a * b                                 # multiply position-by-position
mat = a @ b                                  # dot-product: rows × columns
print(f"a * b (element-wise):\n{elem}")
print(f"  Shape: {elem.shape}")
print(f"a @ b (matrix multiply):\n{mat}")
print(f"  Shape: {mat.shape}")
print()

# --- Why this matters for LLMs ---
print("NOTE: In LLMs, we work with 3D tensors of shape "
      "(batch_size, sequence_length, embedding_dimension)")
print("Example: (8, 512, 768) = 8 sentences, 512 words each, "
      "768 numbers per word.")
