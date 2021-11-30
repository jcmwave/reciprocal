import numpy as np
from enum import Enum, auto


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
    INTERIOR = auto() # Points with no special symmetry

class PointSymmetry(Enum):
    C_INF = auto() # Infinite discrete rotational symmetry (Gamma point only)
    SIGMA_D = auto() # Diagonal mirror symmetry
    SIGMA_H = auto() # Horizontal mirror symmetry
    SIGMA_V = auto() # Vertical mirror symmetry
    C1 = auto() # 2 pi rotational symmetry
    C2 = auto() # pi rotational symmetry
    C3 = auto() # 2/3 pi rotational symmetry
    C4 = auto() # pi /2 rotational symmetry
    C6 = auto() # pi /3 rotational symmetry
    D2 = auto() # combination of C2 and SIGMA_H
    D3 = auto() # combination of C3 and SIGMA_H
    D4 = auto() # combination of C4 and SIGMA_H
    D6 = auto() # combination of C6 and SIGMA_H


def validate_symmetry(symmetry):
    """
    return if symmetry name belongs to set of valid names
    """
    names = [name[0] for name in PointSymmetry.__members__.items()]
    return symmetry in names


class Symmetry(object):
    """
    class for obtaining the number of reflection and rotations for
    a given symmetry type

    Attribues
    ---------
    name: PointSymmetry
        the point symmetry of this symmetry
    """
    #known_symmetries = {'inf', 'XY', 'C2', 'C3', 'C4', 'C6', 'D2', 'D4', 'D6'}

    def __init__(self, symmetry):
        self.group = symmetry

    @classmethod
    def from_string(Symmetry, string):
        if validate_symmetry(string):
            return Symmetry(PointSymmetry[string])
        else:
            raise ValueError("symmetry:{} unknown".format(string))

    def __str__(self):
        return self.group.name

    def __repr__(self):
        return str(self.group)

    def reduce(self):
        reducible = {PointSymmetry.D2: PointSymmetry.C2,
                     PointSymmetry.D3: PointSymmetry.C3,
                     PointSymmetry.D4: PointSymmetry.C4,
                     PointSymmetry.D6: PointSymmetry.C6}

        if not self.group in reducible.keys():
            raise ValueError("cannot reduce symmetry: {}".format(self.group))
        else:
            return Symmetry(reducible[self.group])

    def compatible(self, other):
        """Compares if one symmetries are a multiple of one another


        Parameters
        ----------
        other: Symmetry
            A different Symmetry to compare compatibility to


        Returns
        -------
        bool
            compatibility of symmetries
        """
        if self.group == other.group:
            return True

        #diagonal and horizontal reflections are only compatible with themselves
        functions = [Symmetry.get_n_reflections_y, Symmetry.get_n_reflections_x,
                     Symmetry.get_n_reflections_xy]

        for func in functions:
            if func(self) != func(other):
                return False

        n_rot_self = self.get_n_rotations()
        n_rot_other = other.get_n_rotations()
        if n_rot_other > 1:
            multiple_of_other = n_rot_self % n_rot_other == 0
        else:
            multiple_of_other = False
        if n_rot_self > 1:
            multiple_of_self = n_rot_other % n_rot_self == 0
        else:
            multiple_of_self = False


        if multiple_of_other or multiple_of_self:
            return True
        return False


    def get_n_symmetry_ops(self):
        """
        return total number of symmetry operations
        """
        n_sym_ops = {PointSymmetry.C_INF:np.inf,
                     PointSymmetry.SIGMA_D:1,
                     PointSymmetry.SIGMA_H:1,
                     PointSymmetry.SIGMA_V:1,
                     PointSymmetry.C1:1,
                     PointSymmetry.C2:2,
                     PointSymmetry.C3:3,
                     PointSymmetry.C4:4,
                     PointSymmetry.C6:6,
                     PointSymmetry.D2:4,
                     PointSymmetry.D3:6,
                     PointSymmetry.D4:8,
                     PointSymmetry.D6:12}
        return n_sym_ops[self.group]

    def get_n_rotations(self):
        """
        return number of rotations for the symmetry op
        """
        rotations = {PointSymmetry.C_INF:0,
                     PointSymmetry.SIGMA_D:0,
                     PointSymmetry.SIGMA_H:0,
                     PointSymmetry.SIGMA_V:0,
                     PointSymmetry.C1:1,
                     PointSymmetry.C2:2,
                     PointSymmetry.C3:3,
                     PointSymmetry.C4:4,
                     PointSymmetry.C6:6,
                     PointSymmetry.D2:2,
                     PointSymmetry.D3:3,
                     PointSymmetry.D4:4,
                     PointSymmetry.D6:6}
        return rotations[self.group]

    def get_n_reflections_y(self):
        """
        return number of reflections about the x axis
        """
        reflections = {PointSymmetry.C_INF:0,
                       PointSymmetry.SIGMA_D:0,
                       PointSymmetry.SIGMA_H:1,
                       PointSymmetry.SIGMA_V:0,
                       PointSymmetry.C1:0,
                       PointSymmetry.C2:0,
                       PointSymmetry.C3:0,
                       PointSymmetry.C4:0,
                       PointSymmetry.C6:0,
                       PointSymmetry.D2:0,
                       PointSymmetry.D3:0,
                       PointSymmetry.D4:0,
                       PointSymmetry.D6:0}
        return reflections[self.group]

    def get_n_reflections_x(self):
        """
        return number of reflections about the y axis
        """
        reflections = {PointSymmetry.C_INF:0,
                       PointSymmetry.SIGMA_D:0,
                       PointSymmetry.SIGMA_H:0,
                       PointSymmetry.SIGMA_V:1,
                       PointSymmetry.C1:0,
                       PointSymmetry.C2:0,
                       PointSymmetry.C3:0,
                       PointSymmetry.C4:0,
                       PointSymmetry.C6:0,
                       PointSymmetry.D2:1,
                       PointSymmetry.D3:1,
                       PointSymmetry.D4:1,
                       PointSymmetry.D6:1}
        return reflections[self.group]

    def get_n_reflections_xy(self):
        """
        return number of reflections about the x-y diagonal
        """
        reflections = {PointSymmetry.C_INF:0,
                       PointSymmetry.SIGMA_D:1,
                       PointSymmetry.SIGMA_H:0,
                       PointSymmetry.SIGMA_V:0,
                       PointSymmetry.C1:0,
                       PointSymmetry.C2:0,
                       PointSymmetry.C3:0,
                       PointSymmetry.C4:0,
                       PointSymmetry.C6:0,
                       PointSymmetry.D2:0,
                       PointSymmetry.D3:0,
                       PointSymmetry.D4:0,
                       PointSymmetry.D6:0}
        return reflections[self.group]

    def get_symmetry_cone_angle(self):
        """
        return the smallest angle which  covers a unique segment of the
        brillouin zone
        """
        angle = {PointSymmetry.C1:2.*np.pi/1.0,
                 PointSymmetry.C2:2.*np.pi/2.0,
                 PointSymmetry.C3:2.*np.pi/3.0,
                 PointSymmetry.C4:2.*np.pi/4.0,
                 PointSymmetry.C6:2.*np.pi/6.0,
                 PointSymmetry.D2:2.*np.pi/4.0,
                 PointSymmetry.D3:2.*np.pi/6.0,
                 PointSymmetry.D4:2.*np.pi/8.0,
                 PointSymmetry.D6:2.*np.pi/12.0,
                 PointSymmetry.SIGMA_V:2.*np.pi/2.0}
        if self.group in angle:
            return angle[self.group]
        raise ValueError("symmetry {}".format(self.group) +
                         " has no Symmetry Cone Angle")
