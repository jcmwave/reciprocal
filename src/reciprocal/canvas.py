import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.patches import RegularPolygon, Polygon, Rectangle, Circle
from matplotlib.collections import PatchCollection, RegularPolyCollection
import matplotlib.cm as cm

from lattice import Lattice, ReciprocalLattice

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

    def plot_vectors(self, obj):
        if isinstance(obj, Lattice):
            vectors = obj.vectors
        else:
            vectors = obj

        if vectors.vec1 is None and vectors.vec2 is None:
            vectors.make_vectors()
        plt.sca(self.ax)

        # plt.plot([0.0, vectors.vec1[0]],
        #          [0.0, vectors.vec1[1]],
        #          c=self.colors['vector'])
        # plt.plot([0.0, vectors.vec2[0]],
        #          [0.0, vectors.vec2[1]],
        #          c=self.colors['vector'])
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
        #maxl = np.max([vectors.length1, vectors.length2])
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


    def plot_bzone(self, obj):
        if isinstance(obj, ReciprocalLattice):
            bzone = obj.bzone
        else:
            bzone = obj
        plt.sca(self.ax)
        maxl = bzone.max_extent
        artist = self._get_poly_patch(bzone.vertices,
                                      facecolor=[1.0, 0.0, 0.0, 0.5],
                                      edgecolor=[1.0, 0.0, 0.0, 0.8])
        self.update_bbox([[-maxl, -maxl], [maxl, maxl]])
        self.ax.add_artist(artist)

    def plot_symmetry_points(self, obj):
        if isinstance(obj, ReciprocalLattice):
            sym_points = obj.bzone.symmetry_points
            maxl = obj.bzone.max_extent*0.05
        else:
            sym_points = obj
            maxl = 1.0
        plt.sca(self.ax)
        #maxl = bzone.max_extent
        for point in sym_points:
            artist = Circle(xy=point[1],
                            radius=maxl,
                            facecolor=[1.0, 0.0, 0.0, 0.5],
                            edgecolor=[1.0, 0.0, 0.0, 0.8])
            plt.text(point[1][0], point[1][1], point[0],
                     horizontalalignment='left')

            self.ax.add_artist(artist)


    def plot_ibzone(self, obj):
        if isinstance(obj, ReciprocalLattice):
            ibz = obj.bzone.irreducible
        else:
            ibz = obj
        maxl = ibz.max_extent
        plt.sca(self.ax)
        artist = self._get_poly_patch(ibz.vertices,
                                      facecolor=[0.0, 1.0, 0.0, 0.5],
                                      edgecolor=[0.0, 1.0, 0.0, 0.8],)
        self.update_bbox([[-maxl, -maxl], [maxl, maxl]])
        self.ax.add_artist(artist)



    def plot_primitive(self, obj):
        if isinstance(obj, Lattice):
            primitive = obj.primitive
        else:
            primitive = obj
        maxl = primitive.max_extent
        plt.sca(self.ax)
        #shape = primitive.shape
        #lv1 = np.linalg.norm(primitive.vectors.vec1)
        #lv2 = np.linalg.norm(primitive.vectors.vec2)
        #facecolor = [1.0, 1.0, 1.0, 0.0]
        #edgecolor = [0.0, 0.0, 0.0, 0.3]
        artist = self._get_poly_patch(primitive.vertices,
                                      facecolor=[0.0, 0.0, 1.0, 0.5],
                                      edgecolor=[0.0, 0.0, 1.0, 0.8],)
        # if shape == 'hexagon':
        #     if primitive.alt:
        #         artist = self._get_poly_patch(primitive.polygon)
        #     else:
        #         artist = self._get_hex_patch(primitive.vectors)
        # elif shape in {'rectangle', 'square'}:
        #     artist = self._get_rect_patch(primitive.vectors)
        #maxl = np.max([lv1, lv2])
        self.update_bbox([[-maxl, -maxl], [maxl, maxl]])
        self.ax.add_artist(artist)

    #def plot_irreducible(self, primitive):
    #    poly = self._get_poly_patch(primitive.irreducible_polygon)
    #    self.ax.add_artist(poly)

    @staticmethod
    def increase_bbox(first_orders, second_orders,
                      pos, xi, yi, lv_max, bbox):
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
        # shape = lattice.shape
        # if shape == "hexagon":
        #     patch = self._get_hex_patch(lattice.vectors, pos=pos)
        # elif shape in {'square', 'rectangle'}:
        #     patch = self._get_rect_patch(lattice.vectors, pos=pos)
        # elif shape == 'parallelogram':
        #     patch = self._get_poly_patch(lattice.primitive.polygon,
        #                                  pos=pos)

        patch_generator = self._get_poly_patch
        self._plot_lattice(lattice, patch_generator, orders=orders,
                           label_orders=label_orders)


    def _plot_lattice(self, lattice, patch_generator,
                      orders=None, label_orders=False):
        #shape = lattice.primitive.shape
        vec1 = lattice.vectors.vec1
        vec2 = lattice.vectors.vec2
        lv1 = lattice.vectors.length1
        lv2 = lattice.vectors.length2
        lv_max = np.max([lv1, lv2])
        if orders is None:
            orders = [[-1, 1], [-1, 1]]
        first_orders = range(orders[0][0]-2, orders[0][1]+3)
        second_orders = range(orders[1][0]-2, orders[1][1]+3)
        MOrders = len(first_orders)*len(second_orders)
        v1 = vec1[0:2]
        v2 = vec2[0:2]
        patches = []
        pN = 0
        bbox = copy.copy(self.bbox)
        for xi in first_orders:
            for yi in second_orders:
                pos = v1*xi+v2*yi
                bbox = Canvas.increase_bbox(first_orders, second_orders,
                                            pos, xi, yi, lv_max, bbox)
                patch = patch_generator(lattice.primitive.vertices, pos=pos)
                # if shape == "hexagon":
                #     patch = self._get_hex_patch(lattice.vectors, pos=pos)
                # elif shape in {'square', 'rectangle'}:
                #     patch = self._get_rect_patch(lattice.vectors, pos=pos)
                # elif shape == 'parallelogram':
                #     patch = self._get_poly_patch(lattice.primitive.polygon,
                #                                  pos=pos)
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
            self._plot_primitive_sampling(sampling, color=color)

    def _plot_irreducible_sampling(self, sampling, color=None):
        plt.sca(self.ax)
        if color is None:
            color_set = ['g', 'r', 'b', 'k']
        else:
            color_set = []
            for i in range(4):
                color_set.append(color)
        allowed_keys = {'G':color_set[0], 'K':color_set[1], 'M':color_set[1],
                        'X':color_set[1], 'Y':color_set[1], 'Y1':color_set[1],
                        'Y2':color_set[1], 'C':color_set[1], 'H1':color_set[1],
                        'H2':color_set[1], 'H3':color_set[1],
                        'onSymmetryAxis':color_set[2], 'interior':color_set[3]}

        for key in allowed_keys:
            if key in sampling:
                plt.scatter(sampling[key][:, 0], sampling[key][:, 1],
                            c=allowed_keys[key], zorder=5)

    def _plot_primitive_sampling(self, sampling, color=None, to_plot='all'):
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
        circle_patch = plt.Circle((0, 0), radius=kspace.fermi_radius,
                                  edgecolor='k', linestyle='-',
                                  linewidth=2.0, fill=False)
        self.ax.add_artist(circle_patch)
