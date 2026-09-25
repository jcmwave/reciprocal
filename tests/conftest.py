"""Test configuration that prevents GUI and figure-state side effects."""

import matplotlib

matplotlib.use("Agg", force=True)
