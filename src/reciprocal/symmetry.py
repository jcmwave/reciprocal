import numpy as np
from enum import Enum, auto
from reciprocal.utils import rotation2D, reflection2D, translation2D

class SpecialPoint(Enum):
    GAMMA = auto() #Center of the Brillouin Zone
    K = auto() # Vertex between two edges of a hexagon
    M = auto() # Center of a hexagonal edge
    X = auto() # Center of a cubic/rectangular edge or oblique long edge
    Y = auto() # Center of a rectangular side
    Y1 = auto() # Center of an oblique short edge
    Y2 = auto() # Center of an oblique short edge
    H1 = auto() # Vertex of an oblique Brillouin Zone
    H2 = auto() # Vertex of an oblique Brillouin Zone
    H3 = auto() # Vertex of an oblique Brillouin Zone
    C = auto() # Center of the corner like edge of an oblique Brillouin Zone
    AXIS = auto() # Lying on a Symmetry axis
    EXTERIOR = auto() # Lying on the periodic boundary
    INTERIOR = auto() # Points with no special symmetry

class PointSymmetry(Enum):
    #C_INF = auto() # Infinite discrete rotational symmetry (Gamma point only)
    SIGMA_D = auto() # Diagonal mirror symmetry
    SIGMA_H = auto() # Horizontal mirror symmetry
    SIGMA_V = auto() # Vertical mirror symmetry
    C1 = auto() # 2 pi rotational symmetry
    C2 = auto() # pi rotational symmetry
    C3 = auto() # 2/3 pi rotational symmetry
    C4 = auto() # pi /2 rotational symmetry
    C6 = auto() # pi /3 rotational symmetry
    #D1 = auto() # combination of C1 and SIGMA_H
    #D2 = auto() # combination of C2 and SIGMA_H
    #D3 = auto() # combination of C3 and SIGMA_H
    #D4 = auto() # combination of C4 and SIGMA_H
    #D6 = auto() # combination of C6 and SIGMA_H
    T = auto() # translational symmetry
    #T2 = auto() # twofold translational symmetry

ALIASES = {'D2': [PointSymmetry.SIGMA_H, PointSymmetry.C2], # combination of C2 and SIGMA_H
           'D3': [PointSymmetry.SIGMA_H, PointSymmetry.C3], # combination of C3 and SIGMA_H
           'D4': [PointSymmetry.SIGMA_H, PointSymmetry.C4], # combination of C4 and SIGMA_H
           'D6': [PointSymmetry.SIGMA_H, PointSymmetry.C6]} # combination of C6 and SIGMA_H

def validate_symmetry(symmetry):
    """
    return if symmetry name belongs to set of valid names
    """
    names = [name[0] for name in PointSymmetry.__members__.items()]
    return symmetry in names

def symmetry_from_type(symmetry):
    reflections = [PointSymmetry.SIGMA_D, PointSymmetry.SIGMA_V, PointSymmetry.SIGMA_H]
    rotations = [PointSymmetry.C1, PointSymmetry.C2, PointSymmetry.C3,
                 PointSymmetry.C4, PointSymmetry.C6]
    translations = [PointSymmetry.T]
    if symmetry in reflections:
        return Reflection(symmetry)
    elif symmetry in rotations:
        return Rotation(symmetry)
    elif symmetry in translations:
        return Translation(symmetry)
    else:
        raise KeyError("unknown symmetry of type: {}".format(symmetry))

def symmetry_from_alias(symmetry):
    reflections = [PointSymmetry.SIGMA_D, PointSymmetry.SIGMA_V, PointSymmetry.SIGMA_H]
    rotations = [PointSymmetry.C1, PointSymmetry.C2, PointSymmetry.C3,
                 PointSymmetry.C4, PointSymmetry.C6]
    translations = [PointSymmetry.T]
    if symmetry in reflections:
        return Reflection(symmetry)
    elif symmetry in rotations:
        return Rotation(symmetry)
    elif symmetry in translations:
        return Translation(symmetry)
    else:
        raise KeyError("unknown symmetry of type: {}".format(symmetry))

def symmetry_from_alias(string):
    stack = ALIASES[string]
    return SymmetryCombination(stack)

class SymmetryCombination(object):
    """
    class for applying a combination of symmetries

    Attribues
    ---------
    name: PointSymmetry
        the point symmetry of this symmetry
    """

    def __init__(self, symmetries):
        stack = []
        for symmetry in symmetries:
            if isinstance(symmetry, Symmetry):
                stack.append(symmetry)
            elif isinstance(symmetry, PointSymmetry):
                symmetry_class = symmetry_from_type(symmetry)
                stack.append(symmetry_class)
            else:
                raise TypeError("symmetries for symmetry combination must be of type"+
                                " {} or {},  not {}".format(Symmetry, PointSymmetry, type(symmetry)))
        self.stack = stack

    def get_n_symmetry_ops(self):
        n_ops = 1
        for symmetry in self.stack:
            n_ops *= symmetry.get_n_symmetry_ops()
        return n_ops

    def get_symmetry_cone_angle(self):
        n_ops = 1
        for symmetry in self.stack:
            if not isinstance(symmetry, Translation):
                n_ops *= symmetry.get_n_symmetry_ops()
        return 2*np.pi/n_ops

    def apply_symmetry_operators(self, points, values=None):
        points = np.atleast_2d(points)
        for i_sym, symmetry in enumerate(self.stack + [1.]):
            if i_sym == 0:
                outputs = symmetry.apply_symmetry_operators(points, values=values)
            else:
                if isinstance(outputs, tuple):
                    all_points = outputs[0]
                    all_values = outputs[1]
                else:
                    all_points = outputs
                    all_values = None
                if i_sym == len(self.stack):
                    break
                outputs = symmetry.apply_symmetry_operators(all_points, values=all_values)

        if values is not None:
            return all_points, all_values
        else:
            return all_points

    def __str__(self):
        name = "("
        for sym in self.stack:
            name += str(sym)
            name += ", "
        name = name[:-2] + ")"
        return name

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if isinstance(other, Symmetry):
            return SymmetryCombination(self.stack + [other])
        elif isinstance(other, SymmetryCombination):
            return SymmetryCombination(self.stack + other.stack)
        else:
            raise TypeError("cannot add object of type {} ".format(type(other)) +
                            " to symmetry combination")

    def compatible(self, other):
        if isinstance(other, SymmetryCombination):
            compatible = True
            for sym, other_sym in zip(self.stack, other.stack):
                compatible = sym.compatible(other_sym)
                if not compatible:
                    return False
            return True
        elif isinstance(other, Symmetry):
            compatible = False
            for sym in self.stack:
                compatible = sym.compatible(other)
                if compatible:
                    return True
            return False
        else:
            return False


    def __sub__(self, other):
        if isinstance(other, SymmetryCombination):
            if not len(self.stack) == len(other.stack):
                raise ValueError("stack lengths must be equal for subtraction")
            new_stack = []
            for item1, item2 in zip(self.stack, other.stack):
                new_stack.append(item1-item2)
            return SymmetryCombination(new_stack)
        elif isinstance(other, Symmetry):
            new_stack = []
            for symmetry in self.stack:
                if isinstance(symmetry, type(other)):
                    new_stack.append(symmetry-other)
                else:
                    new_stack.append(symmetry)
            return SymmetryCombination(new_stack)


        else:
            raise TypeError("subtraction valid for SymmetryCombination or Symmetry, not {}".format(type(other)))





class Symmetry(object):
    """
    class for obtaining the number of reflection,rotations translations for
    a given symmetry type

    Attribues
    ---------
    name: PointSymmetry
        the point symmetry of this symmetry
    """
    #known_symmetries = {'inf', 'XY', 'C2', 'C3', 'C4', 'C6', 'D1', 'D2', 'D4', 'D6', 'T', 'T2'}

    def __init__(self, symmetry):
        self.group = symmetry

    def __add__(self, other):
        return SymmetryCombination([self, other])

    @classmethod
    def from_string(Symmetry, string):
        if validate_symmetry(string):
            return symmetry_from_type(PointSymmetry[string])
        elif string in ALIASES:
            return symmetry_from_alias(string)
        else:
            raise ValueError("symmetry:{} unknown".format(string))

    def get_n_symmetry_ops(self):
        op_map = {Rotation:Rotation.get_n_rotations,
                  Reflection:Reflection.get_n_reflections,
                  Translation:Translation.get_n_translations}
        return op_map[type(self)](self)

    def __str__(self):
        return self.group.name

    def __repr__(self):
        return str(self.group)


class Rotation(Symmetry):
    """
    class for definining rotation symmetries
    """

    def __init__(self, *args, **kwargs):
        super(Rotation, self).__init__(*args, **kwargs)

    def get_n_rotations(self):
        """
        return number of rotations for the symmetry op
        """
        rotations = {PointSymmetry.C1:1,
                     PointSymmetry.C2:2,
                     PointSymmetry.C3:3,
                     PointSymmetry.C4:4,
                     PointSymmetry.C6:6}
        return rotations[self.group]

    @classmethod
    def from_n_rotations(Rotation, n_rot):
        rotations = {1:PointSymmetry.C1,
                     2:PointSymmetry.C2,
                     3:PointSymmetry.C3,
                     4:PointSymmetry.C4,
                     6:PointSymmetry.C6}
        return Rotation(rotations[n_rot])

    def apply_symmetry_operators(self, points, values=None):
        n_rot = self.get_n_rotations()
        operators = []
        for i in range(0, n_rot):
            operators.append(rotation2D(i*360./n_rot))

        new_points = []
        points = np.atleast_2d(points)
        for row in range(points.shape[0]):
            point = points[row, :]
            for op in operators:
                new_points.append(op.dot(point))
        new_points = np.vstack(new_points)
        if values is not None:
            new_values = []
            values = np.atleast_2d(values)
            if values.shape[1] == 1:
                new_values = np.repeat(values, n_rot)
                new_values = new_values.reshape( (new_values.size, 1))
            else:
                for row in range(values.shape[0]):
                    value = values[row, :]
                    for op in operators:
                        new_values.append(op.dot(value))
                new_values = np.vstack(new_values)
            return new_points, new_values
        else:
            return new_points

    def get_symmetry_cone_angle(self):
        return 2*np.pi/self.get_n_rotations()

    def compatible(self, other):
        if not isinstance(self, type(other)):
            return False

        n_rot_self = self.get_n_rotations()
        n_rot_other = other.get_n_rotations()
        if n_rot_other > 0:
            multiple_of_other = n_rot_self % n_rot_other == 0
        else:
            multiple_of_other = False
        if n_rot_self > 0:
            multiple_of_self = n_rot_other % n_rot_self == 0
        else:
            multiple_of_self = False
        if multiple_of_other or multiple_of_self:
            return True
        return False

    def __sub__(self, other):
        if not self.compatible(other):
            raise ValueError("cannot subtract rotations if not multiples of each other")
        n_rot_self = self.get_n_rotations()
        n_rot_other = other.get_n_rotations()
        n_rot_new = n_rot_self/n_rot_other
        return Rotation.from_n_rotations(n_rot_new)





class Reflection(Symmetry):
    """
    class for definining reflection symmetries
    """

    def __init__(self, *args, **kwargs):
        super(Reflection, self).__init__(*args, **kwargs)
        if self.group == PointSymmetry.SIGMA_D:
            self.axis = 'xy'
        elif self.group == PointSymmetry.SIGMA_H:
            self.axis = 'x'
        elif self.group == PointSymmetry.SIGMA_V:
            self.axis = 'y'

    def get_n_reflections(self):
        """
        return number of reflections for the symmetry op
        """
        reflections = {PointSymmetry.SIGMA_D:1,
                       PointSymmetry.SIGMA_V:1,
                       PointSymmetry.SIGMA_H:1}
        return reflections[self.group] + 1

    def apply_symmetry_operators(self, points, values=None):
        operators = []
        operators.append(reflection2D(self.axis))

        new_points = []
        points = np.atleast_2d(points)
        for row in range(points.shape[0]):
            point = points[row, :]
            new_points.append(point)
            for op in operators:
                new_points.append(op.dot(point))
        new_points = np.vstack(new_points)
        if values is not None:
            values = np.atleast_2d(values)
            if values.shape[1]:
                new_values = values.repeat(2)
                new_values = new_values.reshape((new_values.size, 1))
            else:
                new_values = []
                for row in range(values.shape[0]):
                    value = values[row, :]
                    values.append(value)
                    for op in operators:
                        new_values.append(op.dot(value))
                new_values = np.vstack(new_values)
            return new_points, new_values
        else:
            return new_points

    def get_symmetry_cone_angle(self):
        return 2*np.pi/self.get_n_reflections()

    def compatible(self, other):
        if not isinstance(self, type(other)):
            return False
        if not self.axis  == other.axis:
            return False
        return True

    def __sub__(self, other):
        if not self.compatible(other):
            raise ValueError("cannot subtract reflections with different planes")
        n_refl_self = self.get_n_reflections()-1
        n_refl_other = other.get_n_reflections()-1
        n_refl_new = n_refl_self-n_refl_other
        if n_refl_new < 0:
            raise ValueError("number of reflections negative")
        if n_refl_new == 1:
            return Reflection(self.group)
        elif n_refl_new ==0:
            return Rotation.from_n_rotations(1)



class Translation(Symmetry):
    """
    class for definining translation symmetries
    """

    def __init__(self, *args, **kwargs):
        super(Translation, self).__init__(*args, **kwargs)
        self.vector1 = ([1., 0.])
        self.vector2 = ([0., 1.])

    def get_n_translations(self):
        """
        return number of translations for the symmetry op inside the BZ
        """
        raise NotImplemented("n translations undefined")
        #translations = {PointSymmetry.T:1}
        #return translations[self.group]

    def apply_symmetry_operators(self, points, n=1, return_orders=False, values=None):
        operators = []

        range1 = np.arange(-n,n+1,1)
        range2 = np.arange(-n,n+1,1)

        new_points = []
        points = np.atleast_2d(points)
        for row in range(points.shape[0]):
            point = points[row, :]
            point[2] = 1.0
            for n1 in range1:
                for n2 in range2:
                    op = translation2D(self.vector1*n1).dot(translation2D(self.vector2*n2))
                    new_points.append(op.dot(point))
        new_points = np.vstack(new_points)
        return_list = [new_points]
        if values is not None:
            new_values = np.repeat(values, int(new_points.size/point.size), axis=0)
            return_list += [new_values]
        if return_orders is True:
            N1, N2 = np.meshgrid(range1, range2)
            return_list += [N1.flatten(), N2.flatten()]
        if len(return_list) == 1:
            return return_list[0]
        else:
            return return_list
        # if return_orders:
        #     N1, N2 = np.meshgrid(range1, range2)
        #     return new_points, N1.flatten(), N2.flatten()
        # else:
        #     return new_points

    def get_symmetry_cone_angle(self):
        raise NotImplemented("symmetry cone angle undefined for translational symmetry")

    def compatible(self, other):
        if not isinstance(self, type(other)):
            return False
        if (np.isclose(np.cross(self.vector1, other.vector1), 0.) and
            np.isclose(np.cross(self.vector2, other.vector2), 0.)):
            #vectors are colinear
            return True
        return False

    def __sub__(self, other):
        raise NotImplemented("subtraction of translation symmetry undefined")
        if not self.compatible(other):
            raise ValueError("cannot subtract translations that are not compatible")
        n_trans_self = self.get_n_translations()-1
        n_trans_other = other.get_n_reflections()-1
        n_trans_new = n_trans_self-n_trans_other
        if n_trans_new < 0:
            raise ValueError("number of translations negative")
        if n_trans_new == 1:
            t = Translation(self.group)
            t.vector = self.vector
            return t
        elif n_trans_new ==0:
            return Rotation.from_n_rotations(1)

    # @classmethod
    # def from_operator_values(Symmetry, op_values):
    #     #if op_values[3] == 1 and np.any(np.array(op_values)[:3]>0):
    #     #    op_values[3] = 0

    #     for symmetry_name in PointSymmetry:
    #         symmetry = Symmetry(symmetry_name)
    #         refl_y = symmetry.get_n_reflections_y()
    #         refl_x = symmetry.get_n_reflections_x()
    #         refl_xy = symmetry.get_n_reflections_xy()
    #         rot = symmetry.get_n_rotations()
    #         this_values = [refl_y, refl_x, refl_xy, rot]
    #         if np.all(this_values == op_values):
    #             return symmetry
    #     if np.sum(np.abs(np.array(op_values))) == 0:
    #         return Symmetry.from_string("C1")
    #     raise ValueError("symmetry could not be determined from operator values: {}".format(op_values))


    # def reduce(self):
    #     reducible = {PointSymmetry.D1: PointSymmetry.C1,
    #                  PointSymmetry.D2: PointSymmetry.C2,
    #                  PointSymmetry.D3: PointSymmetry.C3,
    #                  PointSymmetry.D4: PointSymmetry.C4,
    #                  PointSymmetry.D6: PointSymmetry.C6}

    #     if not self.group in reducible.keys():
    #         raise ValueError("cannot reduce symmetry: {}".format(self.group))
    #     else:
    #         return Symmetry(reducible[self.group])

    # def compatible(self, other):
    #     """Compares if one symmetries are a multiple of one another


    #     Parameters
    #     ----------
    #     other: Symmetry
    #         A different Symmetry to compare compatibility to


    #     Returns
    #     -------
    #     bool
    #         compatibility of symmetries
    #     """
    #     if self.group == other.group:
    #         return True

    #     #diagonal and horizontal reflections are only compatible with themselves
    #     functions = [Symmetry.get_n_reflections_y, Symmetry.get_n_reflections_x,
    #                  Symmetry.get_n_reflections_xy]

    #     for func in functions:
    #         if func(self) != func(other):
    #             return False

    #     n_rot_self = self.get_n_rotations()
    #     n_rot_other = other.get_n_rotations()
    #     if n_rot_other > 1:
    #         multiple_of_other = n_rot_self % n_rot_other == 0
    #     else:
    #         multiple_of_other = False
    #     if n_rot_self > 1:
    #         multiple_of_self = n_rot_other % n_rot_self == 0
    #     else:
    #         multiple_of_self = False


    #     if multiple_of_other or multiple_of_self:
    #         return True
    #     return False

    # def __sub__(self, other):
    #     #if not self.compatible(other):
    #     #    raise ValueError("symmetry cannot be subtracted as they are not compatible"+
    #     #                     " {} - {}".format(self, other))
    #     n_refl_y_new = self.get_n_reflections_y()-other.get_n_reflections_y()
    #     n_refl_x_new = self.get_n_reflections_x()-other.get_n_reflections_x()
    #     n_refl_xy_new = self.get_n_reflections_xy()-other.get_n_reflections_xy()
    #     n_rot_new = self.get_n_rotations()/other.get_n_rotations()
    #     if not np.floor(n_rot_new) == np.ceil(n_rot_new):
    #         raise ValueError("subtraction of symmetries requires rotations to be a multiple of each other")
    #     n_rot_new = int(n_rot_new)
    #     ops = [n_refl_y_new, n_refl_x_new, n_refl_xy_new, n_rot_new]
    #     for new_value in ops:
    #         if new_value < 0:
    #             raise ValueError("subtract leading to negative symmetry undefined")

    #     return Symmetry.from_operator_values(ops)

    # def get_n_symmetry_ops(self):
    #     """
    #     return total number of symmetry operations
    #     """
    #     n_sym_ops = {PointSymmetry.SIGMA_D:1,
    #                  PointSymmetry.SIGMA_H:1,
    #                  PointSymmetry.SIGMA_V:1,
    #                  PointSymmetry.C1:1,
    #                  PointSymmetry.C2:2,
    #                  PointSymmetry.C3:3,
    #                  PointSymmetry.C4:4,
    #                  PointSymmetry.C6:6,
    #                  PointSymmetry.D1:2,
    #                  PointSymmetry.D2:4,
    #                  PointSymmetry.D3:6,
    #                  PointSymmetry.D4:8,
    #                  PointSymmetry.D6:12,
    #                  PointSymmetry:T:1,
    #                  PointSymmetry:T2:2}
    #     return n_sym_ops[self.group]

    # def get_n_rotations(self):
    #     """
    #     return number of rotations for the symmetry op
    #     """
    #     rotations = {PointSymmetry.SIGMA_D:0,
    #                  PointSymmetry.SIGMA_H:0,
    #                  PointSymmetry.SIGMA_V:0,
    #                  PointSymmetry.C1:1,
    #                  PointSymmetry.C2:2,
    #                  PointSymmetry.C3:3,
    #                  PointSymmetry.C4:4,
    #                  PointSymmetry.C6:6,
    #                  PointSymmetry.D1:1,
    #                  PointSymmetry.D2:2,
    #                  PointSymmetry.D3:3,
    #                  PointSymmetry.D4:4,
    #                  PointSymmetry.D6:6,
    #                  PointSymmetry:T:0,
    #                  PointSymmetry:T2:0}
    #     return rotations[self.group]

    # def get_n_reflections_y(self):
    #     """
    #     return number of reflections about the x axis
    #     """
    #     reflections = {PointSymmetry.SIGMA_D:0,
    #                    PointSymmetry.SIGMA_H:1,
    #                    PointSymmetry.SIGMA_V:0,
    #                    PointSymmetry.C1:0,
    #                    PointSymmetry.C2:0,
    #                    PointSymmetry.C3:0,
    #                    PointSymmetry.C4:0,
    #                    PointSymmetry.C6:0,
    #                    PointSymmetry.D1:0,
    #                    PointSymmetry.D2:0,
    #                    PointSymmetry.D3:0,
    #                    PointSymmetry.D4:0,
    #                    PointSymmetry.D6:0}
    #     return reflections[self.group]

    # def get_n_reflections_x(self):
    #     """
    #     return number of reflections about the y axis
    #     """
    #     reflections = {PointSymmetry.SIGMA_D:0,
    #                    PointSymmetry.SIGMA_H:0,
    #                    PointSymmetry.SIGMA_V:1,
    #                    PointSymmetry.C1:0,
    #                    PointSymmetry.C2:0,
    #                    PointSymmetry.C3:0,
    #                    PointSymmetry.C4:0,
    #                    PointSymmetry.C6:0,
    #                    PointSymmetry.D1:1,
    #                    PointSymmetry.D2:1,
    #                    PointSymmetry.D3:1,
    #                    PointSymmetry.D4:1,
    #                    PointSymmetry.D6:1,
    #                    PointSymmetry:T:0,
    #                    PointSymmetry:T2:0}
    #     return reflections[self.group]

    # def get_n_reflections_xy(self):
    #     """
    #     return number of reflections about the x-y diagonal
    #     """
    #     reflections = {PointSymmetry.SIGMA_D:1,
    #                    PointSymmetry.SIGMA_H:0,
    #                    PointSymmetry.SIGMA_V:0,
    #                    PointSymmetry.C1:0,
    #                    PointSymmetry.C2:0,
    #                    PointSymmetry.C3:0,
    #                    PointSymmetry.C4:0,
    #                    PointSymmetry.C6:0,
    #                    PointSymmetry.D1:0,
    #                    PointSymmetry.D2:0,
    #                    PointSymmetry.D3:0,
    #                    PointSymmetry.D4:0,
    #                    PointSymmetry.D6:0,
    #                    PointSymmetry:T:0,
    #                    PointSymmetry:T2:0}
    #     return reflections[self.group]

    # def get_n_translations(self):
    #     """
    #     return number of translations along lattice vectors
    #     """
    #     translations = {PointSymmetry.SIGMA_D:0,
    #                    PointSymmetry.SIGMA_H:0,
    #                    PointSymmetry.SIGMA_V:0,
    #                    PointSymmetry.C1:0,
    #                    PointSymmetry.C2:0,
    #                    PointSymmetry.C3:0,
    #                    PointSymmetry.C4:0,
    #                    PointSymmetry.C6:0,
    #                    PointSymmetry.D1:0,
    #                    PointSymmetry.D2:0,
    #                    PointSymmetry.D3:0,
    #                    PointSymmetry.D4:0,
    #                    PointSymmetry.D6:0,
    #                    PointSymmetry:T:1,
    #                    PointSymmetry:T2:2}
    #     return translations[self.group]


    # def get_symmetry_cone_angle(self):
    #     """
    #     return the smallest angle which  covers a unique segment of the
    #     brillouin zone
    #     """
    #     angle = {PointSymmetry.C1:2.*np.pi/1.0,
    #              PointSymmetry.C2:2.*np.pi/2.0,
    #              PointSymmetry.C3:2.*np.pi/3.0,
    #              PointSymmetry.C4:2.*np.pi/4.0,
    #              PointSymmetry.C6:2.*np.pi/6.0,
    #              PointSymmetry.D1:2.*np.pi/2.0,
    #              PointSymmetry.D2:2.*np.pi/4.0,
    #              PointSymmetry.D3:2.*np.pi/6.0,
    #              PointSymmetry.D4:2.*np.pi/8.0,
    #              PointSymmetry.D6:2.*np.pi/12.0,
    #              PointSymmetry.SIGMA_V:2.*np.pi/2.0}
    #     if self.group in angle:
    #         return angle[self.group]
    #     raise ValueError("symmetry {}".format(self.group) +
    #                      " has no Symmetry Cone Angle")
