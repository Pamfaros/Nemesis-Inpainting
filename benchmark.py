
import argparse
import statistics
import time

import torch

from model.registry import load_model


def benchmark(weights, size, device):

    device = torch.device(device)
    model = load_model(weights, device)

    rgb = torch.rand(
        1, 3, size, size,
        device=device
    )

    mask = torch.zeros(
        1, 1, size, size,
        device=device
    )

    mask[:, :, size//4:3*size//4,
               size//4:3*size//4] = 1

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    timings = []

    with torch.inference_mode():

        # Warm-up
        for _ in range(10):
            model(rgb, mask)

        synchronize()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(30):

            synchronize()

            start = time.perf_counter()

            model(rgb, mask)

            synchronize()

            timings.append(
                (time.perf_counter() - start) * 1000
            )

    print("Device:", device)
    print("Resolution:", size)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    print("Mean ms:", statistics.mean(timings))
    print("Median ms:", statistics.median(timings))

    print("P95 ms:", sorted(timings)[
        int(0.95 * len(timings) + 0.999) - 1
    ])

    if device.type == "cuda":

        print(
            "Peak allocated CUDA memory MiB:",
            torch.cuda.max_memory_allocated(device) / 1024**2
        )


if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("weights")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cuda")

    args = p.parse_args()

    benchmark(
        args.weights,
        args.size,
        args.device
    )
