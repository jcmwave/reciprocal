import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import copy
from matplotlib.patches import RegularPolygon, Polygon, Rectangle, Circle, Wedge
from matplotlib.collections import PatchCollection, RegularPolyCollection
import matplotlib.cm as cm
import scipy.spatial
from reciprocal.lattice import Lattice, LatticeVectors
from reciprocal.unit_cell import UnitCell
from reciprocal.symmetry import SpecialPoint

def choose_color(item, n_items):
    """
    Return a matplotlib color given a total number of colors to draw from.

    Parameters
    ----------
    item: int
        the nth color
    n_items: int
        the total amount of colors to draw from

    Returns
    -------
    (1,4) <np.double> np.array
        the rgb + alpha information
    """
    if n_items <= 10:
        cmap = cm.get_cmap('tab10')
        NColors =10.0
    elif n_items > 10 and n_items <=20:
        cmap = cm.get_cmap('tab20')
        NColors =20.0
    else:
        cmap = cm.get_cmap('turbo')
        NColors = float(n_items)
    color = np.zeros((1,4))
    color[0,:] = cmap((float(item))/NColors)
    return color

class Canvas():

    def __init__(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
            self.fig = fig
        self.ax = ax
        #self.fig
        self.bbox = [[-0., -0.], [0., 0.]]
        ax.set_aspect('equal', 'box')
        self.colors = {}
        self.make_colors()

    def make_colors(self):
        self.colors['vector'] = 'g'

    def update_bbox(self, box):
        for i in range(2):
            for j in range(2):
                if i == 0:
                    if box[i][j] < self.bbox[i][j]:
                        self.bbox[i][j] = box[i][j]
                if i ==1:
                    if box[i][j] > self.bbox[i][j]:
                        self.bbox[i][j] = box[i][j]
        plt.xlim([self.bbox[0][0], self.bbox[1][0]])
        plt.ylim([self.bbox[0][1], self.bbox[1][1]])

    def plot_vectors(self, plot_obj):
        """
        plot the lattice vectors

        Parameters
        ----------
        plt_obj: LatticeVectors or Lattice
            source of vectors to plot

        Returns
        -------
        None
        """
        if isinstance(plot_obj, Lattice):
            vectors = plot_obj.vectors
        elif isinstance(plot_obj, LatticeVectors):
            vectors = plot_obj
        else:
            raise ValueError("cannot plot vectors from object of type :" +
                             "{}".format(type(plot_obj)) +
                             ", required types:"+
                             "{}, ".format(Lattice)+
                             "{}".format(LatticeVectors))

        if vectors.vec1 is None and vectors.vec2 is None:
            vectors.make_vectors()
        plt.sca(self.ax)

        maxl = np.max([vectors.length1, vectors.length2])
        hw = 0.1*maxl
        width = 0.01*maxl
        plt.arrow(0.0, 0.0, vectors.vec1[0], vectors.vec1[1],
                  head_width=hw,
                  width=width,
                  color=self.colors['vector'],
                  ec=self.colors['vector'],
                  length_includes_head=True)
        plt.arrow(0.0, 0.0, vectors.vec2[0], vectors.vec2[1],
                  head_width=hw,
                  width=width,
                  color=self.colors['vector'],
                  ec =self.colors['vector'],
                  length_includes_head=True)

        self.update_bbox([[-maxl, -maxl], [maxl, maxl]])



    def _get_hex_patch(self, vectors,
                       facecolor=[1.0, 1.0, 1.0, 0.0],
                       edgecolor=[0.0, 0.0, 0.0, 0.3],
                       pos=np.zeros(2)):
        lv1 = np.linalg.norm(vectors.vec1)
        lv2 = np.linalg.norm(vectors.vec2)
        polygon_radius = lv1*0.5/np.cos(np.pi/6.0)
        orientation = np.arctan2(vectors.vec1[1],
                                 vectors.vec1[0])
        return RegularPolygon(xy=pos,
                              numVertices=6,
                              radius=polygon_radius,
                              orientation=orientation,
                              facecolor=facecolor,
                              edgecolor=edgecolor
                              )

    def _get_rect_patch(self, vectors,
                        facecolor=[1.0, 1.0, 1.0, 0.0],
                        edgecolor=[0.0, 0.0, 0.0, 0.3],
                        pos=np.zeros(2)):
        lv1 = np.linalg.norm(vectors.vec1)
        lv2 = np.linalg.norm(vectors.vec2)
        return Rectangle(xy=pos-np.array([0.5*lv1, 0.5*lv2]),
                         width=lv1,
                         height=lv2,
                         facecolor=facecolor,
                         edgecolor=edgecolor
                         )

    def _get_poly_patch(self, vertices,
                        facecolor=[1.0, 1.0, 1.0, 0.0],
                        edgecolor=[0.0, 0.0, 0.0, 0.3],
                        pos=np.zeros(2)):
        #for vertex in range(vertices.shape[0]):
        #    vertex += pos
        return Polygon(vertices[:,:2]+pos,
                        closed=True,
                        facecolor=facecolor,
                        edgecolor=edgecolor)

    def _get_point_patch(self, vertices,
                         facecolor=[0.0, 0.0, 0.0, 1.0],
                         edgecolor=[0.0, 0.0, 0.0, 1.0],
                         pos=np.zeros(2)):
        maxl = 0.0
        for row in range(vertices.shape[0]):
            vertex = vertices[row,:]
            norm = np.linalg.norm(vertex)
            if norm > maxl:
                maxl = norm
        #lv1 = np.linalg.norm(vectors.vec1)
        #lv2 = np.linalg.norm(vectors.vec2)
        #maxl = np.max([lv1, lv2])
        radius = maxl*0.05
        return Circle(xy=pos, radius=radius,
                      facecolor=facecolor,
                      edgecolor=edgecolor)


    def plot_bzone(self, plot_obj):
        if isinstance(plot_obj, Lattice):
            bzone = plot_obj.bzone
        else:
            bzone = plot_obj
        plt.sca(self.ax)
        maxl = bzone.max_extent
        artist = self._get_poly_patch(bzone.vertices,
                                      facecolor=[1.0, 0.0, 0.0, 0.5],
                                      edgecolor=[1.0, 0.0, 0.0, 0.8])
        self.update_bbox([[-maxl, -maxl], [maxl, maxl]])
        self.ax.add_artist(artist)

    def plot_special_points(self, plot_obj):
        sym_points = None
        try:
            sym_points = plot_obj.unit_cell.special_points
            maxl = plot_obj.unit_cell.max_extent*0.05
        except:
            pass
        try:
            sym_points = plot_obj.special_points
            maxl = plot_obj.max_extent*0.05
        except:
            pass
        if sym_points == None:
            raise ValueError("cannot plot symmetry points from object of type :" +
                             "{}".format(type(plot_obj)) +
                             ", required types:"+
                             "{}, ".format(Lattice)+
                             "{}".format(UnitCell))
        plt.sca(self.ax)
        #maxl = bzone.max_extent
        for name, pos in sym_points.items():
            artist = Circle(xy=pos[:2],
                            radius=maxl,
                            facecolor=[1.0, 0.0, 0.0, 0.5],
                            edgecolor=[1.0, 0.0, 0.0, 0.8])
            plt.text(pos[0], pos[1], name.name,
                     horizontalalignment='left')

            self.ax.add_artist(artist)


    def plot_ibzone(self, plot_obj):
        if isinstance(plot_obj, Lattice):
            ibz = plot_obj.bzone.irreducible
        else:
            ibz = plot_obj
        maxl = ibz.max_extent
        plt.sca(self.ax)
        artist = self._get_poly_patch(ibz.vertices,
                                      facecolor=[0.0, 1.0, 0.0, 0.5],
                                      edgecolor=[0.0, 1.0, 0.0, 0.8],)
        self.update_bbox([[-maxl, -maxl], [maxl, maxl]])
        self.ax.add_artist(artist)


    def plot_irreducible_uc(self, plot_obj):
        """
        plot the lattice unit cell

        Parameters
        ----------
        plot_obj: UnitCell or Lattice
            source of unit cell to plot

        Returns
        -------
        None
        """
        unit_cell = None
        try:
            unit_cell = plot_obj.unit_cell
        except:
            pass
        if unit_cell is None:
            try:
                unit_cell = plot_obj
            except:
                pass
        if unit_cell is None:
            raise ValueError("cannot plot unit cell from object of type :" +
                             "{}".format(type(plot_obj)) +
                             ", required types:"+
                             "{}, ".format(Lattice)+
                             "{}".format(UnitCell))

        maxl = unit_cell.max_extent
        plt.sca(self.ax)
        artist = self._get_poly_patch(unit_cell.irreducible,
                                      facecolor=[0.0, 0.0, 1.0, 0.5],
                                      edgecolor=[0.0, 0.0, 1.0, 0.8],)
        self.update_bbox([[-maxl, -maxl], [maxl, maxl]])
        self.ax.add_artist(artist)

    def plot_unit_cell(self, plot_obj):
        """
        plot the lattice unit cell

        Parameters
        ----------
        plot_obj: UnitCell or Lattice
            source of unit cell to plot

        Returns
        -------
        None
        """
        unit_cell = None
        try:
            unit_cell = plot_obj.unit_cell
        except:
            pass
        if unit_cell is None:
            try:
                unit_cell = plot_obj
            except:
                pass
        if unit_cell is None:
            raise ValueError("cannot plot unit cell from object of type :" +
                             "{}".format(type(plot_obj)) +
                             ", required types:"+
                             "{}, ".format(Lattice)+
                             "{}".format(UnitCell))

        maxl = unit_cell.max_extent
        plt.sca(self.ax)
        artist = self._get_poly_patch(unit_cell.vertices,
                                      facecolor=[0.0, 0.0, 1.0, 0.5],
                                      edgecolor=[0.0, 0.0, 1.0, 0.8],)
        self.update_bbox([[-maxl, -maxl], [maxl, maxl]])
        self.ax.add_artist(artist)

    @staticmethod
    def increase_bbox(first_orders, second_orders,
                      pos, xi, yi, lv_max, bbox):
        """
        increase the bounding box to fit a lattice

        Parameters
        ----------
        lv_max: float
            longest lattice vector length
        bbox: (2,2) list
            a bounding box for the canvas
        pos: (2,) <np.float> np.array
            a lattice position


        """
        if not (xi <= first_orders[1] or xi >= first_orders[-2] or
                yi <= second_orders[1] or yi >= second_orders[-2]):
            low_x = pos[0]-lv_max*0.6
            high_x = pos[0]+lv_max*0.6
            if low_x < bbox[0][0]:
                bbox[0][0] = low_x
            if high_x > bbox[1][0]:
                bbox[1][0] = high_x
            low_y = pos[1]-lv_max*0.6
            high_y = pos[1]+lv_max*0.6
            if low_y < bbox[0][1]:
                bbox[0][1] = low_y
            if high_y > bbox[1][1]:
                bbox[1][1] = high_y
        return bbox

    def plot_lattice(self, lattice, orders=None, label_orders=False):
        patch_generator = self._get_point_patch
        self._plot_lattice(lattice, patch_generator, orders=orders,
                           label_orders=label_orders)

    def plot_tesselation(self, lattice, orders=None, label_orders=False):
        patch_generator = self._get_poly_patch
        self._plot_lattice(lattice, patch_generator, orders=orders,
                           label_orders=label_orders)

    def plot_lattice_distance_groups(self, lattice, max_order=2, label_orders=False):
        patch_generator = self._get_poly_patch
        groups, distances = lattice.orders_by_distance(max_order)
        n_groups = len(groups)
        for ig, group in enumerate(groups):
            color = choose_color(ig, n_groups).flatten()
            #print(ig, color)
            self._plot_lattice(lattice, patch_generator, orders=group,
                               label_orders=label_orders, facecolor=color,
                               increase_bbox=False)

    def _default_orders(self):
        orders = [[-2, 2], [-2, 2]]
        first_orders = range(orders[0][0]-2, orders[0][1]+3)
        second_orders = range(orders[1][0]-2, orders[1][1]+3)
        order_list = []
        for xi in first_orders:
            for yi in second_orders:
                order_list.append([xi, yi])
        order_array = np.array(order_list)
        return order_array

    def _plot_lattice(self, lattice, patch_generator,
                      orders=None, label_orders=False,
                      facecolor=[1., 1., 1., 0.],
                      increase_bbox=True):
        #shape = lattice.unit_cell.shape
        vec1 = lattice.vectors.vec1
        vec2 = lattice.vectors.vec2
        lv1 = lattice.vectors.length1
        lv2 = lattice.vectors.length2
        lv_max = np.max([lv1, lv2])
        if orders is None:
            orders = self._default_orders()

        v1 = vec1[0:2]
        v2 = vec2[0:2]
        patches = []
        pN = 0
        bbox = copy.copy(self.bbox)
        first_orders = range( np.min(orders[:,0]), np.max(orders[:,0]))
        second_orders = range( np.min(orders[:,1]), np.max(orders[:,1]))
        for row in range(orders.shape[0]):

            xi = orders[row, 0]
            yi = orders[row, 1]
            pos = v1*xi+v2*yi
            if increase_bbox:
                bbox = Canvas.increase_bbox(first_orders, second_orders,
                                            pos, xi, yi, lv_max, bbox)

            patch = patch_generator(lattice.unit_cell.vertices, pos=pos,
                                    facecolor=facecolor)
            patches.append(patch)
            if label_orders:
                Canvas.plot_order(pos, xi, yi)
            pN += 1

        pc = PatchCollection(patches, match_original=True)
        self.ax.add_collection(pc)
        #self.ax.autoscale()
        self.update_bbox(bbox)

    def plot_sampling(self, sampling, color=None):
        if isinstance(sampling, dict):
            self._plot_irreducible_sampling(sampling, color=color)
        else:
            self._plot_unit_cell_sampling(sampling, color=color)

    def _plot_irreducible_sampling(self, sampling, color=None):
        plt.sca(self.ax)
        if color is None:
            color_set = ['g', 'r', 'b', 'k']
        else:
            color_set = []
            for i in range(4):
                color_set.append(color)
        color_mapping = {}
        for spoint in SpecialPoint:
            if spoint is SpecialPoint.GAMMA:
                color_mapping[spoint] = color_set[0]
            elif spoint is SpecialPoint.AXIS:
                color_mapping[spoint] = color_set[2]
            elif spoint is SpecialPoint.INTERIOR:
                color_mapping[spoint] = color_set[3]
            else:
                color_mapping[spoint] = color_set[1]
        """
        allowed_keys = {'G':color_set[0], 'K':color_set[1], 'M':color_set[1],
                        'X':color_set[1], 'Y':color_set[1], 'Y1':color_set[1],
                        'Y2':color_set[1], 'C':color_set[1], 'H1':color_set[1],
                        'H2':color_set[1], 'H3':color_set[1],
                        'onSymmetryAxis':color_set[2], 'interior':color_set[3]}
        """
        for key in color_mapping:
            if key in sampling:
                plt.scatter(sampling[key][:, 0], sampling[key][:, 1],
                            c=color_mapping[key], zorder=5)

    def _plot_unit_cell_sampling(self, sampling, color=None, to_plot='all'):
        plt.sca(self.ax)
        n_points = sampling.shape[0]
        if color is None:
            if n_points <= 10:
                cmap = cm.get_cmap('tab10')
                n_colors = 10.0
            elif  10 < n_points <= 20:
                cmap = cm.get_cmap('tab20')
                n_colors = 20.0
            elif 20 < n_points <= 40:
                cmap = cm.get_cmap('tab20c')
                n_colors = 40.0
            else:
                cmap = cm.get_cmap('jet')
                n_colors = float(n_points)
            colors = cmap((np.arange(n_points)/n_colors))
        else:
            colors = color
        if to_plot == 'all':
            plt.scatter(sampling[:, 0], sampling[:, 1], c=colors, zorder=5)
        else:
            plt.scatter(sampling[0:to_plot, 0],
                        sampling[0:to_plot, 1],
                        c=colors)

    @staticmethod
    def plot_order(pos, order1, order2):
        plt.text(pos[0], pos[1], '({},{})'.format(order1, order2),
                 horizontalalignment='center',
                 verticalalignment='center',
                 clip_on=True)

    def plot_fermi_circle(self, kspace):
        if kspace.fermi_radius is None:
            raise ValueError("cannot plot fermi circle: fermi radius not set")
        radius = kspace.fermi_radius
        circle_patch = Circle((0, 0), radius=radius,
                                  edgecolor='k', linestyle='-',
                                  linewidth=2.0, fill=False)
        self.ax.add_artist(circle_patch)
        self.update_bbox([[-radius, -radius], [radius, radius]])

    def plot_symmetry_cone(self, kspace):
        if kspace.fermi_radius is None:
            raise ValueError("cannot plot symmetry cone: fermi radius not set")
        if kspace.symmetry is None:
            raise ValueError("cannot plot symmetry cone: symmetry not set")
        wedge_patch = Wedge((0, 0), kspace.fermi_radius,
                             0., np.degrees(kspace.symmetry.get_symmetry_cone_angle()),
                                  edgecolor='k', linestyle='--',
                                  linewidth=2.0, fill=False)
        self.ax.add_artist(wedge_patch)

    def plot_bloch_families(self, bloch_families, plot_n_families='all',
                           legend=False):
        """
        plot the bloch families

        Parameters
        ----------
        plot_n_families: int or str
            how many families to plot
        legend: bool
            display legend

        Returns
        -------
        None
        """

        if plot_n_families=='all':
            plot_n_families = len(bloch_families)

        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        plt.sca(self.ax)
        n_families = 0
        print("plot_n_families: {}".format(plot_n_families))
        for family_number, family in sorted(bloch_families.items()):
            #print(family_number)
            #print(family)
            if n_families >= plot_n_families:
                return
            #print("plot family")
            color = choose_color(family_number, plot_n_families)
            label = "Bloch Family {}".format(family_number+1)
            try:
                plt.scatter(family[:,0], family[:,1], color=color, label=label)
            except TypeError:
                plt.scatter(family.kx, family.ky, color=color, label=label)
            n_families += 1
        if legend:
            plt.legend(bbox_to_anchor=[1.01,0.99], loc='upper left')

    def plot_point_sampling(self, points, plot_n_points='all', color=None):
        """
        plot a point sampling

        Parameters
        ----------
        plot_n_points: int or str
            how many points to plot
        legend: bool
            display legend

        Returns
        -------
        None
        """

        if plot_n_points=='all':
            try:
                plot_n_points = len(points.shape[0])
            except AttributeError:
                plot_n_points = points.n_rows

        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        plt.sca(self.ax)
        for point_number in range(plot_n_points):
            try:
                point = points[point_number, :]
            except TypeError:
                point = points.k[point_number, :]
            if color is None:
                point_color = choose_color(point_number, plot_n_points)
            else:
                point_color = color
            #label = " {}".format(family_number+1)
            plt.scatter(point[0], point[1], color=point_color)
        #if legend:
        #    plt.legend(bbox_to_anchor=[1.01,0.99], loc='upper left')

    def plot_interpolation(self, kpoints, values):
        plt.sca(self.ax)
        xi = kpoints.k[:,[0,1]]
        tessellation = scipy.spatial.Delaunay(xi)
        triangles = tessellation.vertices
        plt.tripcolor(xi[:,0],xi[:,1],triangles,values,shading='flat',
                      cmap='magma',edgecolors='g')
