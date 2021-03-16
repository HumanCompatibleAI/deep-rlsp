import pickle

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def main():
    n_samples = 1
    skill = "balancing"
    loss_files = [
        f"discriminator_cheetah_{skill}_{n_samples}_gail.pkl",
        f"discriminator_cheetah_{skill}_{n_samples}_rlsp.pkl",
        f"discriminator_cheetah_{skill}_{n_samples}_average_features.pkl",
        f"discriminator_cheetah_{skill}_{n_samples}_waypoints.pkl",
    ]
    labels = ["GAIL", "Deep RLSP", "AverageFeatures", "Waypoints"]
    outfile = f"{skill}_{n_samples}.pdf"
    moving_average_n = 10

    sns.set_context("paper", font_scale=3.6, rc={"lines.linewidth": 3})
    sns.set_style("white")
    matplotlib.rc(
        "font",
        **{
            "family": "serif",
            "serif": ["Computer Modern"],
            "sans-serif": ["Latin Modern"],
        },
    )
    matplotlib.rc("text", usetex=True)

    markers_every = 100
    markersize = 10

    colors = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#e41a1c",
        "#dede00",
        "#999999",
        "#f781bf",
        "#a65628",
        "#984ea3",
    ]

    markers = ["o", "^", "s", "v", "d", "+", "x", "."]

    L = len(loss_files)
    colors = colors[:L]
    markers = markers[:L]

    # plt.figure(figsize=(20, 6))
    for loss_file, label, color, marker in zip(loss_files, labels, colors, markers):
        print(loss_file)
        with open(loss_file, "rb") as f:
            losses = pickle.load(f)
        print(losses.shape)

        if moving_average_n > 1:
            losses = moving_average(losses, moving_average_n)

        plt.plot(
            range(len(losses)),
            losses,
            label=label,
            color=color,
            marker=marker,
            markevery=markers_every,
            markersize=markersize,
        )
        plt.xlabel("iterations")
        plt.ylabel("cross-entropy loss")

    # plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    # plt.show()


if __name__ == "__main__":
    main()
