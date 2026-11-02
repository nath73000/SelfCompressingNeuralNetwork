"""Plot the run saved by examples/mnist.py (light + dark PNG for the README) and print the results table."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ASSETS_DIR  = ROOT / "assets"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink_secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "baseline": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink_secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "baseline": "#383835", "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


def load_run():
    with open(RESULTS_DIR / "mnist_config.json") as f:
        config = json.load(f)
    with open(RESULTS_DIR / "mnist_metrics.csv", newline="") as f:
        rows = [{k: float(v) for k, v in row.items() if v != ""} for row in csv.DictReader(f)]
    return config, rows


def percent_of_float32(row, config):
    return 100 * row["num_bits"] / (32 * config["initial_num_param"])


def style_axis(ax, theme, title):
    ax.set_facecolor(theme["surface"])
    ax.set_title(title, loc="left", fontsize=11, color=theme["ink"], pad=10)
    ax.grid(axis="y", color=theme["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["baseline"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.set_xlabel("Epoch", fontsize=9, color=theme["muted"])


def end_dot(ax, x, y, color, theme):
    """End marker in the series color, with a ring in the surface color"""
    ax.plot([x], [y], "o", markersize=8, color=color, markeredgecolor=theme["surface"], markeredgewidth=2, zorder=3)


def end_label(ax, x, y, text, color, theme):
    end_dot(ax, x, y, color, theme)
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10, color=theme["ink"])


def plot(config, rows, theme_name):
    theme  = THEMES[theme_name]
    series = theme["series"]
    epochs = [r["epoch"] for r in rows]
    last   = epochs[-1]
    final  = rows[-1]
    num_hidden_layers = len(config["hidden_shape"]) - 1   # the output layer is never pruned

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharex=True)
    fig.patch.set_facecolor(theme["surface"])

    # --- Accuracy (epoch 0 is the untrained model: left out so it doesn't flatten the curve) ---
    ax = axes[0]
    trained = [r for r in rows if r["epoch"] >= 1]
    ax.plot([r["epoch"] for r in trained], [r["accuracy"] for r in trained], color=series[0], linewidth=2)
    end_label(ax, last, final["accuracy"], f'{final["accuracy"]:.2f}%', series[0], theme)
    style_axis(ax, theme, "Test accuracy (%)")

    # --- Model size ---
    ax = axes[1]
    ax.plot(epochs, [percent_of_float32(r, config) for r in rows], color=series[0], linewidth=2)
    end_label(ax, last, percent_of_float32(final, config), f"{percent_of_float32(final, config):.1f}%", series[0], theme)
    ax.set_ylim(bottom=0)
    style_axis(ax, theme, "Weights size (% of float32)")

    # --- Neurons per hidden layer ---
    ax = axes[2]
    for i in range(num_hidden_layers):
        neurones = [r[f"neurones_layer_{i}"] for r in rows]
        ax.plot(epochs, neurones, color=series[i], linewidth=2, label=f"Hidden layer {i + 1}: {neurones[-1]:.0f} / {neurones[0]:.0f} kept")
        end_dot(ax, last, neurones[-1], series[i], theme)
    ax.set_ylim(bottom=0)
    style_axis(ax, theme, "Neurons per hidden layer")
    ax.legend(frameon=False, fontsize=9, labelcolor=theme["ink_secondary"], loc="upper right")
    ax.text(config["pruning_after_epoch"] + 3, 0.12, "pruning starts", transform=ax.get_xaxis_transform(),
            ha="left", va="center", fontsize=9, color=theme["muted"])

    for ax in axes:
        ax.axvline(config["pruning_after_epoch"], color=theme["baseline"], linewidth=1, zorder=0)
    axes[0].set_xlim(0, last * 1.2)   # room for the end labels

    model = "784-" + "-".join(str(h) for h in config["hidden_shape"])
    fig.text(0.01, 0.97, "Self-compressing MLP on MNIST", fontsize=13, fontweight="semibold", color=theme["ink"], va="top")
    fig.text(0.01, 0.90, f'MLP {model} · γ = {config["gamma"]} · {config["epochs"]} epochs · seed {config["seed"]}',
             fontsize=10, color=theme["ink_secondary"], va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=3)
    return fig


def print_table(config, rows):
    final       = rows[-1]
    float32_kb  = 32 * config["initial_num_param"] / 8 / 1024
    size_kb     = final["num_bits"] / 8 / 1024
    compression = 32 * config["initial_num_param"] / final["num_bits"]
    num_hidden_layers = len(config["hidden_shape"]) - 1
    kept  = ", ".join(f'{config["hidden_shape"][i]} → {final[f"neurones_layer_{i}"]:.0f}' for i in range(num_hidden_layers))
    model = "MLP 784-" + "-".join(str(h) for h in config["hidden_shape"])

    print("| Dataset | Model | Test accuracy | Weights size | Compression | Avg. bits / kept weight | Hidden neurons kept |")
    print("|---|---|---|---|---|---|---|")
    print(f'| MNIST | {model} | {final["accuracy"]:.2f}% | {size_kb:.1f} KB (float32: {float32_kb:.0f} KB) '
          f'| ×{compression:.1f} | {final["bits_per_weight"]:.2f} | {kept} |')


def main():
    config, rows = load_run()
    ASSETS_DIR.mkdir(exist_ok=True)
    for theme_name, suffix in (("light", ""), ("dark", "_dark")):
        fig = plot(config, rows, theme_name)
        fig.savefig(ASSETS_DIR / f"mnist_training{suffix}.png", dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
    print_table(config, rows)


if __name__ == "__main__":
    main()
