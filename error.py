import torch
import torch.nn as nn

# Dummy model that breaks the gradient chain
class BrokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        with torch.no_grad():  # ❌ breaks gradient flow
            x = self.linear(x)
        return x

model = BrokenModel()
x = torch.randn(1, 4, requires_grad=True)  # ✅ x requires grad
out = model(x)  # ❌ no grad_fn because model used no_grad
loss = out.mean()
print("loss:", loss)

loss.backward()  # 💥 raises: element 0 of tensors does not require grad and does not have a grad_fn
