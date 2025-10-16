import torch
from thop import profile, clever_format
import time
import copy



def count_params_flops_thop(model, B, H, W):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_size = (B, 3, H, W)

    dummy_input = torch.randn(input_size).to(device)
    model_copy = copy.deepcopy(model).eval()


    with torch.no_grad():
        # FLOPs and params
        flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
    #flops, params = clever_format([flops, params], "%.3f")
    print(f"Model: {model.__class__.__name__}")
    print(f"{input_size}")
    print(f"Params: {params / 1e6:.3f}M ")
    print(f"FLOPs : {flops/ 1e9:.3f}G")

    return params, flops


def measure_fps(model, B=1, H=224, W=224, num_warmup=10, num_iters=50, device="cuda"):
    """
    Measure inference FPS of a PyTorch model.

    Args:
        model: torch.nn.Module
        inputs: example input tensor (B, C, H, W)
        num_warmup: number of warm-up runs (ignored in timing)
        num_iters: number of iterations to average
        device: "cuda" or "cpu"
    """
    input_size = (B, 3, H, W)
    model = model.to(device).eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warm-up (for stable GPU clocks)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timing loop
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    fps = num_iters / elapsed

    print(f'Inference FPS: {fps:.2f} for {num_iters} iterations')

    return fps