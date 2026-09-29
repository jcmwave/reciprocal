"""Matplotlib adapters for immutable reciprocal geometry and sampling data."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon, Wedge
from numpy.typing import ArrayLike

from reciprocal.brillouin_zone import BrillouinZone
from reciprocal.band_path import HighSymmetryPath
from reciprocal.cells.model import PolygonDomain, SamplingResult, UnitCell
from reciprocal.kspace import KSpace
from reciprocal.kvector import KVectorGroup
from reciprocal.lattice import Lattice, LatticeVectors
from reciprocal.spectrum import (
    EvanescentDisk,
    ExpansionMap,
    FieldSamples,
    IntersectionDomain,
    InterpolationResult,
    KDomain,
    KSampling,
    PeriodicKSpace,
    PeriodicSamplingPlan,
    PolygonKDomain,
    PropagationDisk,
    PupilDomain,
)
from reciprocal.symmetry import SpecialPoint


def choose_color(item: int, n_items: int) -> np.ndarray:
    """Return a deterministic RGBA color for an item in a collection."""

    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if n_items <= 10:
        color_map, count = mpl.colormaps["tab10"], 10.0
    elif n_items <= 20:
        color_map, count = mpl.colormaps["tab20"], 20.0
    else:
        color_map, count = mpl.colormaps["turbo"], float(n_items)
    return np.asarray(color_map(float(item) / count)).reshape(1, 4)


def _generate_orders(order_limits: ArrayLike) -> np.ndarray:
    limits = np.asarray(order_limits, dtype=int)
    if limits.shape != (2, 2):
        raise ValueError("order limits must have shape (2, 2)")
    first = range(int(limits[0, 0]) - 2, int(limits[0, 1]) + 3)
    second = range(int(limits[1, 0]) - 2, int(limits[1, 1]) + 3)
    return np.asarray([(i, j) for i in first for j in second], dtype=int)


def _as_points(value: object) -> np.ndarray:
    """Return an ``(N, 2)`` view from every supported sampling container."""

    if isinstance(value, FieldSamples):
        value = value.sampling
    if isinstance(value, PeriodicSamplingPlan):
        value = value.target
    if isinstance(value, ExpansionMap):
        value = value.target
    if isinstance(value, (KSampling, SamplingResult, InterpolationResult)):
        array = value.points
    elif isinstance(value, KVectorGroup):
        array = value.k
    else:
        array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError("point data must have shape (N, 2) or (N, 3)")
    transverse = np.asarray(array[:, :2], dtype=complex)
    if not np.all(np.isfinite(transverse)) or np.any(np.abs(np.imag(transverse)) > 0.0):
        raise ValueError("transverse point coordinates must be finite and real")
    return np.asarray(np.real(transverse), dtype=float)


def _sampling_weights(value: object, *, physical: bool) -> np.ndarray | None:
    if isinstance(value, FieldSamples):
        value = value.sampling
    if isinstance(value, PeriodicSamplingPlan):
        value = value.target
    if isinstance(value, ExpansionMap):
        value = value.target
    if isinstance(value, KSampling):
        return value.physical_weights if physical else value.normalized_weights
    if isinstance(value, SamplingResult):
        return value.weights * value.integration_element if physical else value.weights
    if isinstance(value, KVectorGroup):
        weighting = value.weighting
        if np.all(np.isfinite(weighting)):
            return weighting
    return None


def _cell_for_lattice(lattice: Lattice) -> UnitCell:
    if lattice.lattice_type == "reciprocal":
        return lattice.brillouin_zone.cell
    return lattice.primitive_cell


def _zone(value: object) -> BrillouinZone:
    if isinstance(value, BrillouinZone):
        return value
    if isinstance(value, PeriodicSamplingPlan):
        return value.zone
    if isinstance(value, PeriodicKSpace):
        return value.direct_lattice.make_reciprocal().brillouin_zone
    if isinstance(value, Lattice):
        lattice = value if value.lattice_type == "reciprocal" else value.make_reciprocal()
        return lattice.brillouin_zone
    raise TypeError("expected a Lattice, BrillouinZone, PeriodicKSpace, or sampling plan")


def _cell(value: object) -> UnitCell:
    if isinstance(value, UnitCell):
        return value
    if isinstance(value, BrillouinZone):
        return value.cell
    if isinstance(value, Lattice):
        return _cell_for_lattice(value)
    if isinstance(value, PeriodicSamplingPlan):
        return value.zone.cell
    if isinstance(value, PeriodicKSpace):
        return value.direct_lattice.make_reciprocal().brillouin_zone.cell
    raise TypeError("expected a Lattice, UnitCell, BrillouinZone, or sampling plan")


class Canvas:
    """Draw reciprocal geometry into a caller-owned Matplotlib axes."""

    def __init__(self, ax=None):
        if ax is None:
            self.fig, ax = plt.subplots(1, 1)
        else:
            self.fig = ax.figure
        self.ax = ax
        self.ax.set_aspect("equal", "box")
        self.bbox = [[0.0, 0.0], [0.0, 0.0]]
        self.colors = {"vector": "g"}

    def make_colors(self) -> None:
        self.colors = {"vector": "g"}

    def update_bbox(self, box: ArrayLike) -> None:
        bounds = np.asarray(box, dtype=float)
        if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)):
            raise ValueError("box must contain finite lower and upper coordinate pairs")
        self.bbox[0][0] = min(self.bbox[0][0], float(bounds[0, 0]))
        self.bbox[0][1] = min(self.bbox[0][1], float(bounds[0, 1]))
        self.bbox[1][0] = max(self.bbox[1][0], float(bounds[1, 0]))
        self.bbox[1][1] = max(self.bbox[1][1], float(bounds[1, 1]))
        x0, y0 = self.bbox[0]
        x1, y1 = self.bbox[1]
        scale = max(x1 - x0, y1 - y0, 1.0)
        if x0 == x1:
            x0, x1 = x0 - 0.05 * scale, x1 + 0.05 * scale
        if y0 == y1:
            y0, y1 = y0 - 0.05 * scale, y1 + 0.05 * scale
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)

    def _update_from_points(self, points: np.ndarray, padding: float = 0.05) -> None:
        if len(points) == 0:
            return
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        scale = max(float(np.max(upper - lower)), float(np.max(np.abs(points))), 1.0)
        margin = padding * scale
        self.update_bbox([lower - margin, upper + margin])

    @staticmethod
    def _polygon_patch(vertices: ArrayLike, **kwargs: Any) -> Polygon:
        points = np.asarray(vertices, dtype=float)
        return Polygon(points[:, :2], closed=True, **kwargs)

    def plot_vectors(self, plot_obj: Lattice | LatticeVectors):
        vectors = plot_obj.vectors if isinstance(plot_obj, Lattice) else plot_obj
        if not isinstance(vectors, LatticeVectors):
            raise TypeError("plot_vectors expects a Lattice or LatticeVectors")
        maximum = max(vectors.length1, vectors.length2)
        artists = []
        for vector in (vectors.vec1, vectors.vec2):
            artists.append(
                self.ax.arrow(
                    0.0,
                    0.0,
                    vector[0],
                    vector[1],
                    head_width=0.1 * maximum,
                    width=0.01 * maximum,
                    color=self.colors["vector"],
                    ec=self.colors["vector"],
                    length_includes_head=True,
                )
            )
        self.update_bbox([[-maximum, -maximum], [maximum, maximum]])
        return tuple(artists)

    def plot_bzone(self, plot_obj: object, **kwargs: Any):
        zone = _zone(plot_obj)
        style = {
            "facecolor": (1.0, 0.0, 0.0, 0.25),
            "edgecolor": (1.0, 0.0, 0.0, 0.9),
            "linewidth": 2.0,
        }
        style.update(kwargs)
        artist = self._polygon_patch(zone.vertices, **style)
        self.ax.add_patch(artist)
        self._update_from_points(zone.vertices[:, :2])
        return artist

    def plot_ibzone(self, plot_obj: object, **kwargs: Any):
        zone = _zone(plot_obj)
        style = {
            "facecolor": (0.0, 1.0, 0.0, 0.25),
            "edgecolor": (0.0, 0.6, 0.0, 0.9),
            "linewidth": 2.0,
        }
        style.update(kwargs)
        artist = self._polygon_patch(zone.irreducible_domain.vertices, **style)
        self.ax.add_patch(artist)
        self._update_from_points(zone.irreducible_domain.vertices[:, :2])
        return artist

    def plot_irreducible_uc(self, plot_obj: object, **kwargs: Any):
        if isinstance(plot_obj, (BrillouinZone, Lattice, PeriodicKSpace, PeriodicSamplingPlan)):
            return self.plot_ibzone(plot_obj, **kwargs)
        vertices = getattr(plot_obj, "irreducible", None)
        if vertices is None:
            raise TypeError("plot_irreducible_uc expects reciprocal geometry")
        style = {
            "facecolor": (0.0, 0.0, 1.0, 0.25),
            "edgecolor": (0.0, 0.0, 1.0, 0.9),
            "linewidth": 2.0,
        }
        style.update(kwargs)
        artist = self._polygon_patch(vertices, **style)
        self.ax.add_patch(artist)
        self._update_from_points(np.asarray(vertices)[:, :2])
        return artist

    def plot_unit_cell(self, plot_obj: object, **kwargs: Any):
        cell = _cell(plot_obj)
        style = {
            "facecolor": (0.0, 0.0, 1.0, 0.2),
            "edgecolor": (0.0, 0.0, 1.0, 0.9),
            "linewidth": 2.0,
        }
        style.update(kwargs)
        artist = self._polygon_patch(cell.vertices, **style)
        self.ax.add_patch(artist)
        self._update_from_points(cell.vertices[:, :2])
        return artist

    def plot_special_points(self, plot_obj: object, **kwargs: Any):
        try:
            zone = _zone(plot_obj)
            special_points = zone.special_points
            scale = zone.max_extent
            new_model = True
        except TypeError:
            special_points = getattr(plot_obj, "special_points", None)
            if special_points is None:
                raise TypeError("object does not expose reciprocal-space special points")
            scale = float(getattr(plot_obj, "max_extent"))
            new_model = False
        radius = kwargs.pop("radius", 0.025 * scale)
        artists = []
        for label, point in special_points.items():
            position = point.cartesian if new_model else np.asarray(point)
            name = str(label) if new_model else getattr(label, "name", str(label))
            artist = Circle(
                position[:2],
                radius=radius,
                facecolor=(1.0, 0.0, 0.0, 0.5),
                edgecolor=(1.0, 0.0, 0.0, 0.9),
                **kwargs,
            )
            self.ax.add_patch(artist)
            self.ax.text(position[0], position[1], name, horizontalalignment="left")
            artists.append(artist)
        return tuple(artists)

    @staticmethod
    def increase_bbox(first_orders, second_orders, pos, xi, yi, lv_max, bbox):
        del first_orders, second_orders, xi, yi
        bbox[0][0] = min(bbox[0][0], pos[0] - 0.6 * lv_max)
        bbox[0][1] = min(bbox[0][1], pos[1] - 0.6 * lv_max)
        bbox[1][0] = max(bbox[1][0], pos[0] + 0.6 * lv_max)
        bbox[1][1] = max(bbox[1][1], pos[1] + 0.6 * lv_max)
        return bbox

    def _default_orders(self) -> np.ndarray:
        return _generate_orders([[-2, 2], [-2, 2]])

    def _plot_lattice(
        self,
        lattice: Lattice,
        *,
        filled: bool,
        orders: ArrayLike | None,
        label_orders: bool,
        facecolor: Any,
        edgecolor: Any,
        linewidth: float,
        increase_bbox: bool,
    ):
        if not isinstance(lattice, Lattice):
            raise TypeError("lattice plotting expects a Lattice")
        order_array = self._default_orders() if orders is None else np.asarray(orders, dtype=int)
        if order_array.ndim != 2 or order_array.shape[1] != 2:
            raise ValueError("orders must have shape (N, 2)")
        positions = order_array @ lattice.vectors.basis[:, :2]
        cell = _cell_for_lattice(lattice)
        patches = []
        labels = []
        extent = max(lattice.vectors.length1, lattice.vectors.length2)
        point_radius = 0.035 * extent
        face_array, edge_array = np.asarray(facecolor), np.asarray(edgecolor)
        for index, (order, position) in enumerate(zip(order_array, positions)):
            current_face = face_array[index] if face_array.ndim == 2 else facecolor
            current_edge = edge_array[index] if edge_array.ndim == 2 else edgecolor
            if filled:
                patch = self._polygon_patch(
                    cell.vertices[:, :2] + position,
                    facecolor=current_face,
                    edgecolor=current_edge,
                    linewidth=linewidth,
                )
            else:
                patch = Circle(
                    position,
                    radius=point_radius,
                    facecolor=current_face,
                    edgecolor=current_edge,
                    linewidth=linewidth,
                )
            patches.append(patch)
            if label_orders:
                labels.append(self.plot_order(position, int(order[0]), int(order[1])))
        collection = PatchCollection(patches, match_original=True)
        self.ax.add_collection(collection)
        if increase_bbox:
            self._update_from_points(positions, padding=0.15)
        return collection, tuple(labels)

    def plot_lattice(self, lattice, orders=None, label_orders=False, **kwargs: Any):
        return self._plot_lattice(
            lattice,
            filled=False,
            orders=orders,
            label_orders=label_orders,
            facecolor=kwargs.pop("facecolor", (0.0, 0.0, 0.0, 1.0)),
            edgecolor=kwargs.pop("edgecolor", (0.0, 0.0, 0.0, 1.0)),
            linewidth=kwargs.pop("lw", kwargs.pop("linewidth", 1.0)),
            increase_bbox=kwargs.pop("increase_bbox", True),
        )

    def plot_tessellation(self, lattice, orders=None, label_orders=False, **kwargs: Any):
        return self._plot_lattice(
            lattice,
            filled=True,
            orders=orders,
            label_orders=label_orders,
            facecolor=kwargs.pop("facecolor", (1.0, 1.0, 1.0, 0.0)),
            edgecolor=kwargs.pop("edgecolor", (0.0, 0.0, 0.3, 0.4)),
            linewidth=kwargs.pop("lw", kwargs.pop("linewidth", 1.5)),
            increase_bbox=kwargs.pop("increase_bbox", True),
        )

    def plot_tesselation(self, lattice, orders=None, label_orders=False, **kwargs: Any):
        warnings.warn(
            "plot_tesselation is deprecated; use plot_tessellation",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.plot_tessellation(
            lattice, orders=orders, label_orders=label_orders, **kwargs
        )

    def plot_lattice_distance_groups(self, lattice, max_order=2, label_orders=False):
        groups, _distances = lattice.orders_by_distance(max_order)
        return tuple(
            self.plot_tessellation(
                lattice,
                orders=group,
                label_orders=label_orders,
                facecolor=choose_color(index, len(groups)).ravel(),
                increase_bbox=False,
            )[0]
            for index, group in enumerate(groups)
        )

    def plot_sampling(self, sampling, color=None):
        if isinstance(sampling, Mapping):
            return self._plot_irreducible_sampling(sampling, color=color)
        return self.plot_point_sampling(sampling, color=color)

    def _plot_irreducible_sampling(self, sampling: Mapping, color=None):
        default = {SpecialPoint.GAMMA: "g", SpecialPoint.AXIS: "b", SpecialPoint.INTERIOR: "k"}
        artists = []
        for key, values in sampling.items():
            points = np.atleast_2d(values)
            selected = color if color is not None else default.get(key, "r")
            artists.append(self.ax.scatter(points[:, 0], points[:, 1], c=selected, zorder=5))
            self._update_from_points(points[:, :2])
        return tuple(artists)

    def _plot_unit_cell_sampling(self, sampling, color=None, to_plot="all"):
        return self.plot_point_sampling(sampling, plot_n_points=to_plot, color=color)

    def plot_order(self, pos, order1, order2):
        return self.ax.text(
            pos[0],
            pos[1],
            f"({order1},{order2})",
            horizontalalignment="center",
            verticalalignment="center",
            clip_on=True,
        )

    def plot_domain(self, domain: object, **kwargs: Any):
        if isinstance(domain, (PropagationDisk, EvanescentDisk, PupilDomain)):
            if isinstance(domain, PropagationDisk):
                radii = (domain.radius,)
            elif isinstance(domain, EvanescentDisk):
                radii = (domain.propagating_radius, domain.max_parallel_wavevector)
            else:
                radii = tuple(
                    radius for radius in (domain.inner_radius, domain.outer_radius) if radius > 0.0
                )
            style = {"fill": False, "linewidth": 2.0, "edgecolor": "k"}
            style.update(kwargs)
            artists = []
            for radius in radii:
                artist = Circle(domain.center, radius=radius, **style)
                self.ax.add_patch(artist)
                artists.append(artist)
            bounds = domain.bounds()
            self.update_bbox([[bounds[0], bounds[2]], [bounds[1], bounds[3]]])
            return tuple(artists)
        if isinstance(domain, UnitCell):
            domain = domain.domain
        if isinstance(domain, PolygonKDomain):
            polygon = domain.polygon
        elif isinstance(domain, PolygonDomain):
            polygon = domain
        else:
            polygon = None
        if polygon is not None:
            style = {"fill": False, "linewidth": 2.0, "edgecolor": "k"}
            style.update(kwargs)
            artist = self._polygon_patch(polygon.vertices, **style)
            self.ax.add_patch(artist)
            self._update_from_points(polygon.vertices[:, :2])
            return (artist,)
        if isinstance(domain, IntersectionDomain):
            return tuple(
                artist
                for member in domain.domains
                for artist in self.plot_domain(member, **kwargs)
            )
        raise TypeError(f"unsupported domain type: {type(domain).__name__}")

    def plot_spectrum(self, value: object, **kwargs: Any):
        if isinstance(value, FieldSamples):
            domain = value.sampling.domain
        elif isinstance(value, KSampling):
            domain = value.domain
        elif isinstance(value, KSpace):
            domain = value.spectrum_domain
        elif isinstance(value, PeriodicSamplingPlan):
            domain = value.target.domain
        elif isinstance(value, ExpansionMap):
            domain = value.target.domain
        else:
            domain = value
        if not isinstance(domain, KDomain):
            raise TypeError("plot_spectrum expects KSpace, KSampling, FieldSamples, or KDomain")
        return self.plot_domain(domain, **kwargs)

    def plot_fermi_circle(self, kspace, linewidth=2.0, color="k", fill=False, **kwargs):
        if isinstance(kspace, KSpace):
            radius = kspace.propagating_radius
            if radius is None:
                raise ValueError("cannot plot propagating circle: spectrum is not configured")
        elif isinstance(kspace, PropagationDisk):
            radius = kspace.radius
        elif isinstance(kspace, EvanescentDisk):
            radius = kspace.propagating_radius
        else:
            radius = float(kspace)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("circle radius must be finite and positive")
        artist = Circle(
            (0.0, 0.0),
            radius=radius,
            linewidth=linewidth,
            edgecolor=color,
            facecolor=color if fill else "none",
            fill=fill,
            **kwargs,
        )
        self.ax.add_patch(artist)
        self.update_bbox([[-radius, -radius], [radius, radius]])
        return artist

    def plot_symmetry_cone(self, value, color="k", start_angle=0.0):
        if isinstance(value, (BrillouinZone, Lattice, PeriodicKSpace, PeriodicSamplingPlan)):
            return self.plot_ibzone(value, fill=False, edgecolor=color, linestyle="--")
        if not isinstance(value, KSpace) or value.propagating_radius is None:
            raise TypeError("plot_symmetry_cone expects periodic geometry or a configured KSpace")
        if value.symmetry is None:
            raise ValueError("cannot plot symmetry cone: symmetry is not set")
        opening = np.degrees(value.symmetry.get_symmetry_cone_angle())
        artist = Wedge(
            (0.0, 0.0),
            value.propagating_radius,
            start_angle,
            start_angle + opening,
            edgecolor=color,
            linestyle="--",
            linewidth=2.0,
            fill=False,
        )
        self.ax.add_patch(artist)
        return artist

    def plot_bloch_families(self, bloch_families, plot_n_families="all", legend=False):
        if isinstance(bloch_families, PeriodicSamplingPlan):
            expansion = bloch_families.expansion
        elif isinstance(bloch_families, ExpansionMap):
            expansion = bloch_families
        else:
            expansion = None
        if expansion is not None:
            grouped: dict[int, list[np.ndarray]] = {}
            for point, generator in zip(expansion.target.points, expansion.primary_generators):
                grouped.setdefault(generator.source_index, []).append(point)
            families: Mapping = grouped
        elif isinstance(bloch_families, Mapping):
            families = bloch_families
        else:
            raise TypeError("plot_bloch_families expects a mapping, ExpansionMap, or plan")
        maximum = len(families) if plot_n_families == "all" else int(plot_n_families)
        artists = []
        for count, (family_number, family) in enumerate(sorted(families.items())):
            if count >= maximum:
                break
            points = _as_points(family)
            artist = self.ax.scatter(
                points[:, 0],
                points[:, 1],
                color=choose_color(count, maximum),
                label=f"Bloch Family {family_number + 1}",
            )
            artists.append(artist)
            self._update_from_points(points)
        if legend:
            self.ax.legend(bbox_to_anchor=(1.01, 0.99), loc="upper left")
        return tuple(artists)

    def plot_point_sampling(
        self,
        points,
        plot_n_points="all",
        color="k",
        marker="o",
        label="",
        **kwargs: Any,
    ):
        coordinates = _as_points(points)
        count = len(coordinates) if plot_n_points == "all" else int(plot_n_points)
        if count < 0:
            raise ValueError("plot_n_points must be non-negative")
        count = min(count, len(coordinates))
        selected = coordinates[:count]
        colors = (
            np.vstack([choose_color(index, max(count, 1)) for index in range(count)])[:, 0, :]
            if color is None and count
            else color
        )
        artist = self.ax.scatter(
            selected[:, 0], selected[:, 1], color=colors, marker=marker, label=label, **kwargs
        )
        self._update_from_points(selected)
        return artist

    def plot_point_sampling_weighted(
        self,
        points,
        weighting=None,
        plot_n_points="all",
        marker="o",
        label="",
        *,
        physical=False,
        **kwargs: Any,
    ):
        coordinates = _as_points(points)
        if weighting is None:
            weights = _sampling_weights(points, physical=physical)
        else:
            weights = np.asarray(weighting, dtype=float)
        if weights is None:
            raise ValueError("weighting is required for data without attached quadrature weights")
        if weights.shape != (len(coordinates),) or not np.all(np.isfinite(weights)):
            raise ValueError("weighting must contain one finite value per point")
        count = len(coordinates) if plot_n_points == "all" else int(plot_n_points)
        count = min(count, len(coordinates))
        coordinates, weights = coordinates[:count], weights[:count]
        norm = mpl.colors.Normalize(vmin=float(np.min(weights)), vmax=float(np.max(weights)))
        artist = self.ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=weights,
            marker=marker,
            label=label,
            norm=norm,
            **kwargs,
        )
        self._update_from_points(coordinates)
        return artist

    def plot_field(
        self,
        field: FieldSamples,
        *,
        component: int | tuple[int, ...] | None = None,
        magnitude: bool = True,
        **kwargs: Any,
    ):
        if not isinstance(field, FieldSamples):
            raise TypeError("plot_field expects FieldSamples")
        values = np.asarray(field.values)
        valid = np.asarray(field.valid, dtype=bool)
        if component is not None:
            indices = (component,) if isinstance(component, int) else component
            values = values[(slice(None),) + tuple(indices)]
        elif values.ndim > 1:
            values = np.linalg.norm(values.reshape(len(values), -1), axis=1)
        if values.ndim != 1:
            raise ValueError("select a scalar component before plotting")
        if np.iscomplexobj(values):
            values = np.abs(values) if magnitude else np.real(values)
        artist = self.ax.scatter(
            field.sampling.points[valid, 0],
            field.sampling.points[valid, 1],
            c=values[valid],
            **kwargs,
        )
        self._update_from_points(field.sampling.points[valid])
        return artist

    def plot_interpolation(self, kpoints, values=None, **kwargs: Any):
        if isinstance(kpoints, InterpolationResult):
            coordinates = kpoints.points[kpoints.valid]
            data = (
                kpoints.values[kpoints.valid]
                if values is None
                else np.asarray(values)[kpoints.valid]
            )
        elif isinstance(kpoints, FieldSamples):
            coordinates = kpoints.sampling.points[kpoints.valid]
            data = (
                kpoints.values[kpoints.valid]
                if values is None
                else np.asarray(values)[kpoints.valid]
            )
        else:
            coordinates = _as_points(kpoints)
            if values is None:
                raise ValueError("values are required unless data is attached to the input")
            data = np.asarray(values)
        if data.ndim != 1 or len(data) != len(coordinates):
            raise ValueError("plot_interpolation requires one scalar value per point")
        if np.iscomplexobj(data):
            data = np.abs(data)
        triangulation = scipy.spatial.Delaunay(coordinates)
        style = {"shading": "flat", "cmap": "magma", "edgecolors": "g"}
        style.update(kwargs)
        artist = self.ax.tripcolor(
            coordinates[:, 0],
            coordinates[:, 1],
            triangulation.simplices,
            data,
            **style,
        )
        self._update_from_points(coordinates)
        return artist

    def plot_high_symmetry_path(self, path: HighSymmetryPath, **kwargs: Any):
        """Draw a high-symmetry path and its labelled nodes."""

        if not isinstance(path, HighSymmetryPath):
            raise TypeError("path must be a HighSymmetryPath")
        style = {"color": "tab:blue", "linewidth": 2.0}
        style.update(kwargs)
        lines = []
        starts = (0, *tuple(int(index) for index in path.break_indices))
        stops = (*tuple(int(index) for index in path.break_indices), len(path.points))
        for start, stop in zip(starts, stops):
            lines.extend(self.ax.plot(path.points[start:stop, 0], path.points[start:stop, 1], **style))
        nodes = self.ax.scatter(
            path.points[path.node_indices, 0],
            path.points[path.node_indices, 1],
            color=style["color"],
            zorder=5,
        )
        texts = []
        for label, index in zip(path.labels, path.node_indices):
            point = path.points[index]
            texts.append(self.ax.text(point[0], point[1], label))
        self._update_from_points(path.points)
        return tuple(lines), nodes, tuple(texts)


__all__ = ["Canvas", "choose_color"]
