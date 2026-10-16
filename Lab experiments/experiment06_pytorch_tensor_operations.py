"""Experiment 6: Create PyTorch tensors and perform arithmetic operations."""
import torch

a = torch.tensor([10.0, 20.0, 30.0])
b = torch.tensor([2.0, 4.0, 5.0])

print("Tensor A:", a)
print("Tensor B:", b)
print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
