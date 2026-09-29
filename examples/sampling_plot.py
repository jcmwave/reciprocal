"""Create a deterministic reciprocal-space sampling plot."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reciprocal import KSpace
from reciprocal.canvas import Canvas
from reciprocal.spectrum import CartesianGrid


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="PNG file to create")
    args = parser.parse_args()

    space = KSpace.propagating(np.pi, refractive_index=1.0)
    sampling = space.sample_pupil(CartesianGrid((41, 41)))

    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    canvas = Canvas(ax)
    canvas.plot_spectrum(space)
    canvas.plot_point_sampling_weighted(sampling, cmap="viridis")
    ax.set(xlabel=r"$k_x$", ylabel=r"$k_y$", title="Reciprocal-space sampling")
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
