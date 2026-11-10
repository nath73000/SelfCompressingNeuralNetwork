import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import time
import csv
import json
from pathlib import Path

from selfcompress import SelfCompressingMLP
from selfcompress import DynamicAdamW
from selfcompress import model_size_bits, float32_size_bits

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def save_metrics(metrics: list, path: Path):
    """Write one row per epoch (rewritten every epoch so an interrupted run keeps its results)"""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[-1].keys()), restval="")  # epoch 0 has no train metrics
        writer.writeheader()
        writer.writerows(metrics)


def main():
    epochs = 200
    batch_size = 256
    hidden_shape = [128, 128, 10]
    init_num_bits = 32
    init_exponent = -32

    lr_weight = 0.001
    lr_quantization = 0.12
    lr_decay_epoch = 180
    pruning_after_epoch = 50

    gamma = 0.02     # Larger -> stronger pressure to use fewer bits

    activation_layer = nn.ReLU

    device = "cpu"

    seed = 13
    torch.manual_seed(seed)


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = DataLoader(MNIST('data', train=True,  download=True, transform=transform), batch_size=batch_size, shuffle=True, drop_last=True)
    test_set  = DataLoader(MNIST('data', train=False, download=True, transform=transform), batch_size=batch_size)

    model = SelfCompressingMLP(
        input_size=784,
        hidden_shape=hidden_shape,
        init_num_bits=init_num_bits,
        init_exponent=init_exponent,
        activation_layer=activation_layer
    ).to(device)

    optimizer = DynamicAdamW(
        model=model,
        lr_weight=lr_weight,
        lr_quantization=lr_quantization
    )


    initial_num_param    = model.get_num_param()
    initial_tot_num_bits = model.count_tot_num_bits()
    float32_bits         = float32_size_bits(model)     # weights + biases in float32: the reference for the model size

    # --- Save the config next to the metrics (used by plot_mnist.py) ---
    RESULTS_DIR.mkdir(exist_ok=True)
    config = {
        "epochs": epochs, "batch_size": batch_size, "hidden_shape": hidden_shape,
        "init_num_bits": init_num_bits, "init_exponent": init_exponent,
        "lr_weight": lr_weight, "lr_quantization": lr_quantization,
        "lr_decay_epoch": lr_decay_epoch, "pruning_after_epoch": pruning_after_epoch,
        "gamma": gamma, "seed": seed,
        "initial_num_param": initial_num_param, "initial_tot_num_bits": initial_tot_num_bits,
        "float32_size_bits": float32_bits,
    }
    with open(RESULTS_DIR / "mnist_config.json", "w") as f:
        json.dump(config, f, indent=4)

    metrics = []

    for epoch in range (epochs):
        start_time = time.perf_counter()
        train_metrics = {}
        if epoch > 0:
            # --- Training ---
            model.train()
            compression_loss_sum    = torch.zeros((), device=device)
            classification_loss_sum = torch.zeros((), device=device)
            num_batches = 0

            for x, y in train_set:
                x, y = x.to(device), y.to(device)

                # Divided by the initial number of weights: gamma keeps the same strength when the network shrinks
                norm_num_bits  = model.get_tot_num_bits() / initial_num_param
                compression_loss = gamma * norm_num_bits

                classification_loss = F.cross_entropy(model(x.reshape(x.shape[0], -1)), y)

                loss = compression_loss + classification_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                compression_loss_sum    += compression_loss.detach()
                classification_loss_sum += classification_loss.detach()
                num_batches += 1

            train_metrics = {
                "compression_loss": compression_loss_sum.item() / num_batches,
                "classification_loss": classification_loss_sum.item() / num_batches
            }

            # --- Pruning ---
            if epoch > pruning_after_epoch:
                del loss, compression_loss, classification_loss, norm_num_bits
                with torch.no_grad():
                    keep_and_cut_indices = model.keep_and_cut_indices()
                    model.prune_MLP(optimizer, keep_and_cut_indices)

        if epoch in (lr_decay_epoch, epochs):
            optimizer.scale_learning_rate(0.5)
                    
        # --- Evaluation ---
        model.eval()
        with torch.inference_mode():
            correct, test_loss_sum = 0, 0.0
            for x, y in test_set:
                x, y = x.to(device).reshape(x.shape[0], -1), y.to(device)
                logits = model(x)
                test_loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
                correct       += (logits.argmax(dim=1) == y).sum().item()
            accuracy  = 100 * correct / len(test_set.dataset)
            test_loss = test_loss_sum / len(test_set.dataset)

        num_bits     = model.count_tot_num_bits()
        num_neurones = model.count_num_neurones_by_layer()
        # Everything needed to store the model: weights on whole bits + biases, bit-widths and exponents
        model_size           = model_size_bits(model)["total_bits"]
        percent_float32_size = 100 * model_size / float32_bits

        if device == "mps":
            torch.mps.synchronize()
        epoch_time = time.perf_counter() - start_time

        metrics.append({
            "epoch": epoch,
            **train_metrics,
            "test_loss": test_loss,
            "accuracy": accuracy,
            "num_bits": num_bits,                                               # weights only, continuous bit-widths: what the loss sees
            "bits_per_weight": num_bits / model.get_num_param(),
            "percent_initial_bits": 100 * num_bits / initial_tot_num_bits,
            "model_size_bits": model_size,
            "percent_float32_size": percent_float32_size,
            **{f"neurones_layer_{i}": n for i, n in enumerate(num_neurones)},   # one column per layer
            "lr_weight":       optimizer.lr_weight,
            "time_per_epoch":  epoch_time,
        })

        print(f'Epoch {epoch:4d} | accuracy {accuracy:6.2f}% | num bits {num_bits:9_.0f} | model size {percent_float32_size:6.2f}% of float32 | {epoch_time:6.2f}s / epoch | {num_neurones}')
        save_metrics(metrics, RESULTS_DIR / "mnist_metrics.csv")



if __name__ == "__main__":
    main()