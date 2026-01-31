import torch
import torch.nn as nn
import numpy as np
import json

# ============================================================
# Model
# ============================================================
class SmallNetwork(nn.Module):
    def __init__(self):
        super(SmallNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(1 * 4 * 4, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


# ============================================================
# Utility: Print Weights & Biases (ASCII)
# ============================================================
def print_weights_biases(model):
    print("\n=== WEIGHTS AND BIASES (ASCII) ===\n")
    for name, param in model.named_parameters():
        print(f"[{name}]")
        arr = param.detach().cpu().numpy()
        if arr.ndim == 1:
            print("Bias:")
            print(" ".join(f"{v:7.4f}" for v in arr))
        else:
            print("Weights:")
            arr2 = arr.reshape(arr.shape[0], -1)
            for row in arr2:
                print(" ".join(f"{v:7.4f}" for v in row))
        print()


# ============================================================
# MODE 1: Layer / Tensor-level importance
# ============================================================
def identify_layer_level_importance(model, images, num_runs=3, device="cuda:1"):
    model = model.to(device)
    images = images.to(device)

    activation_stats = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            activation_stats[name] = {'sum': 0.0, 'count': 0}

            def make_hook(mname):
                def hook(mod, inp, out):
                    val = out[0] if isinstance(out, (tuple, list)) else out
                    if torch.is_tensor(val):
                        activation_stats[mname]['sum'] += float(val.abs().mean())
                        activation_stats[mname]['count'] += 1
                return hook

            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(images)

    for h in hooks:
        h.remove()

    module_act_mean = {
        k: v['sum'] / v['count']
        for k, v in activation_stats.items()
        if v['count'] > 0
    }

    param_scores = {}
    for pname, p in model.named_parameters():
        layer = pname.rpartition('.')[0]
        act_mean = module_act_mean.get(layer, 0.0)
        w_mean = float(p.detach().abs().mean())
        param_scores[pname] = act_mean * (w_mean + 1e-12)

    return param_scores


# ============================================================
# MODE 2: True per-weight importance (Gradient × Weight)
# ============================================================
def identify_per_weight_importance(model, images, num_runs=3, device="cuda:1"):
    model = model.to(device)
    images = images.to(device)
    model.eval()

    weight_importance = {
        name: torch.zeros_like(p, device=device)
        for name, p in model.named_parameters()
    }

    for _ in range(num_runs):
        model.zero_grad(set_to_none=True)

        out = model(images)
        objective = out.abs().mean()
        objective.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                weight_importance[name].add_((p.grad * p).abs())

    for name in weight_importance:
        weight_importance[name].div_(num_runs)

    return weight_importance  # stays on GPU


# ============================================================
# NEW MODE: Individual weight-level importance
# ============================================================
def identify_individual_weight_importance(model, images, num_runs=3, device="cuda"):
    """
    Individual weight-level importance calculation using activation × weight magnitudes.
    """
    model = model.to(device)
    images = images.to(device)
    model.eval()

    module_activation_map = {}
    hooks = []

    def hook_function(name):
        def hook(module, input, output):
            activation_mean = float(output.detach().abs().mean().item())
            module_activation_map[name].append(activation_mean)
        return hook

    # Collect activation means for all modules
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module_activation_map[name] = []
            hooks.append(module.register_forward_hook(hook_function(name)))

    # Perform forward passes
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(images)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    weight_importance = {}
    for name, param in model.named_parameters():
        layer_name = ".".join(name.split(".")[:-1])  # Get parent module name
        if layer_name in module_activation_map:
            activation_mean = np.mean(module_activation_map[layer_name])  # Mean activation for layer
            importance = param.detach().abs() * activation_mean  # Compute individual importance
            weight_importance[name] = importance.cpu().numpy()  # Individual weights stored

    return weight_importance


# ============================================================
# Dispatcher
# ============================================================
def identify_importance(model, images, mode="layer", num_runs=3, device="cuda:1"):
    if mode == "layer":
        return identify_layer_level_importance(model, images, num_runs, device)
    elif mode == "per_weight":
        return identify_per_weight_importance(model, images, num_runs, device)
    elif mode == "individual_weight":
        return identify_individual_weight_importance(model, images, num_runs, device)
    else:
        raise ValueError("mode must be 'layer', 'per_weight', or 'individual_weight'")


# ============================================================
# Explain Top-K Important Weights
# ============================================================

def explain_topk_weights(model, weight_importance, top_k=5):
    print("\n=== DETAILED TOP-K WEIGHT TRACEABILITY ===\n")

    param_dict = dict(model.named_parameters())

    for pname, imp_tensor in weight_importance.items():
        param = param_dict[pname]

        # Convert numpy arrays back to PyTorch tensors if necessary
        if isinstance(imp_tensor, np.ndarray):
            imp_tensor = torch.tensor(imp_tensor)

        flat_imp = imp_tensor.flatten()
        flat_wt = param.flatten()

        if flat_imp.numel() == 0:
            continue

        k = min(top_k, flat_imp.numel())
        topk_vals, topk_idx = torch.topk(flat_imp, k)

        print(f"\nParameter: {pname}")
        print("-" * (len(pname) + 11))

        for rank in range(k):
            idx = topk_idx[rank].item()
            imp_val = topk_vals[rank].item()
            wt_val = flat_wt[idx].item()
            tensor_index = np.unravel_index(idx, param.shape)

            print(
                f"Rank {rank+1}: "
                f"Index {tensor_index} | "
                f"Weight = {wt_val:.6f} | "
                f"Importance = {imp_val:.6f}"
            )

# ============================================================
# RUN EVERYTHING
# ============================================================
if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    model = SmallNetwork()
    dummy_input = torch.randn(4, 3, 8, 8)

    # 1️⃣ Print raw weights
    print_weights_biases(model)

    # 2️⃣ Layer-level importance
    print("\n=== LAYER-LEVEL IMPORTANCE ===")
    layer_scores = identify_importance(
        model, dummy_input, mode="layer", num_runs=3, device=device
    )
    for k, v in layer_scores.items():
        print(f"{k}: {v:.6f}")

    # 3️⃣ Per-weight importance
    print("\n=== PER-WEIGHT IMPORTANCE (Gradient × Weight) ===")
    weight_scores = identify_importance(
        model, dummy_input, mode="per_weight", num_runs=3, device=device
    )
    for k, v in weight_scores.items():
        flat = v.flatten()
        top_vals, _ = torch.topk(flat, min(5, flat.numel()))
        
        if 0:
            print(
                f"{k} | top-5 importance values:",
                top_vals.detach().cpu().numpy()
            )
            
        formatted_vals = " ".join(f"{v:7.4f}" for v in top_vals.detach().cpu().numpy())
        print(f"{k} | top-5 importance values: {formatted_vals}")
        # Alternative detailed print (commented out)            
#                 print(" ".join(f"{v:7.4f}" for v in row))

    # 5️⃣ Exact traceability
    explain_topk_weights(
        model,
        weight_scores,
        top_k=15
    )

    # 4️⃣ Individual weight importance (Activation × Weight Magnitude)
    print("\n=== INDIVIDUAL WEIGHT IMPORTANCE ===")
    individual_importance = identify_importance(
        model, dummy_input, mode="individual_weight", num_runs=3, device=device
    )
    for k, v in individual_importance.items():
        flat = v.flatten()
        top_vals, _ = torch.topk(torch.tensor(flat), min(5, flat.size))
        
        if 0:
            print(
                f"{k} | top-5 importance values:",
                top_vals.numpy()
            )

        formatted_vals = " ".join(f"{v:7.4f}" for v in top_vals.detach().cpu().numpy())
        print(f"{k} | top-5 importance values: {formatted_vals}")

    # 5️⃣ Exact traceability
    explain_topk_weights(
        model,
        individual_importance,
        top_k=15
    )
