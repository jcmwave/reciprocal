"""Create a deterministic reciprocal-space sampling plot."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reciprocal import KSpace
from reciprocal.canvas import Canvas


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="PNG file to create")
    args = parser.parse_args()

    space = KSpace(np.pi, symmetry="D4", fermi_radius=2.0)
    points, _weights = space.regular_sampler.sample(
        grid_type="cartesian",
        constraint={"type": "n_points", "value": 81},
        restrict_to_sym_cone=True,
    )

    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    canvas = Canvas(ax)
    canvas.plot_fermi_circle(space)
    canvas.plot_symmetry_cone(space)
    canvas.plot_point_sampling(points, color="tab:blue")
    ax.set(xlabel=r"$k_x$", ylabel=r"$k_y$", title="Reciprocal-space sampling")
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
