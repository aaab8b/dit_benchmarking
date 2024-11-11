from models import DiT_models
import torch
import torch.nn as nn
from functools import wraps
import torch.nn.functional as F

model = DiT_models['DiT-XL/2'](
        input_size=32,
        num_classes=1000
    )
input_shapes = {}

def capture_matmul_shapes(original_func):
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        # Capture shapes of tensor inputs
        input_shapes[original_func.__name__] = [arg.shape for arg in args if isinstance(arg, torch.Tensor)]
        # Call the original matmul function
        return original_func(*args, **kwargs)
    return wrapper

F.scaled_dot_product_attention = capture_matmul_shapes(F.scaled_dot_product_attention)

# torch.addmm = capture_matmul_shapes(torch.addmm)

feat = torch.zeros((2, 4, 32, 32))
label = torch.zeros((2)).long()
t = torch.randint(0, 1000, (feat.shape[0],))


model(feat, t, label)

# for layer, shape in input_shapes.items():
    # print(f"{layer}: {shape}")

for op_name, shapes in input_shapes.items():
    print(f"{op_name}: {shapes}")