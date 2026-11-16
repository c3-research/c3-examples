#!/usr/bin/env python3
"""PyTorch matrix multiplication benchmark.

Multiplies square matrices of increasing size on the CPU and on the GPU,
prints the timings and saves results/benchmark.json and results/benchmark.png.

Everything it imports is already in the Docker image named in .c3: C3 jobs
have no outbound network, so nothing can be installed while the job runs.
"""

import json
import os
import time

import torch
from PIL import Image, ImageDraw

SIZES = [1024, 2048, 4096]
CPU_REPEATS = 2
GPU_REPEATS = 20


def time_matmul(size: int, device: torch.device, repeats: int) -> float:
    """Return the best wall-clock time in seconds for one size x size matmul."""
    generator = torch.Generator(device=device).manual_seed(size)
    a = torch.rand(size, size, device=device, generator=generator)
    b = torch.rand(size, size, device=device, generator=generator)
    torch.matmul(a, b)  # warm-up: loads libraries and allocates memory
    best = float("inf")
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        torch.matmul(a, b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return best


def save_plot(rows: list[dict], gpu_name: str, path: str) -> None:
    """Draw GPU speed-up per matrix size as a bar chart."""
    width, height, margin, base = 640, 400, 60, 330
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 20), f"Matrix multiplication: {gpu_name} speed-up over CPU", fill="black")
    draw.line((margin, base, width - margin, base), fill="black")
    top = max(row["speedup"] for row in rows)
    slot = (width - 2 * margin) / len(rows)
    for index, row in enumerate(rows):
        left = margin + index * slot + slot * 0.2
        right = margin + (index + 1) * slot - slot * 0.2
        bar_top = base - (base - 80) * row["speedup"] / top
        draw.rectangle((left, bar_top, right, base), fill="#4ECDC4", outline="black")
        draw.text((left, bar_top - 16), f"{row['speedup']:.0f}x", fill="black")
        draw.text((left, base + 8), f"{row['size']} x {row['size']}", fill="black")
    image.save(path)


def main() -> None:
    print("PyTorch matrix multiplication benchmark")
    print("=" * 56)
    print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. Check `hardware:` and `docker.requires_accelerator` in .c3.")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print()
    print(f"{'size':>12} {'cpu (s)':>10} {'gpu (s)':>10} {'speed-up':>10}")

    rows = []
    for size in SIZES:
        cpu = time_matmul(size, torch.device("cpu"), CPU_REPEATS)
        gpu = time_matmul(size, torch.device("cuda"), GPU_REPEATS)
        rows.append({"size": size, "cpu_seconds": cpu, "gpu_seconds": gpu, "speedup": cpu / gpu})
        print(f"{f'{size} x {size}':>12} {cpu:>10.4f} {gpu:>10.5f} {cpu / gpu:>9.0f}x")

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as handle:
        json.dump({"gpu": gpu_name, "torch": torch.__version__, "results": rows}, handle, indent=2)
    save_plot(rows, gpu_name, "results/benchmark.png")
    print()
    print("Saved results/benchmark.json and results/benchmark.png")


if __name__ == "__main__":
    main()
