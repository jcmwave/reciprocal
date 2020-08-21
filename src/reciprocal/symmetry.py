import numpy as np


def validate_symmetry(symmetry):
    """
    return if symmetry name belongs to set of valid names
    """
    syms = Symmetry.known_symmetries
    return symmetry in syms

class Symmetry(object):
    """
    class for obtaining the number of reflection and rotations for
    a given symmetry type
    """


    known_symmetries = {'inf', 'XY', 'C2', 'C3', 'C4', 'C6', 'D2', 'D4', 'D6'}

    def __init__(self, symmetry):
        if validate_symmetry(symmetry):
            self.name = symmetry
        else:
            raise ValueError("symmetry:{} unknown".format(symmetry))

    def get_n_rotations(self):
        """
        return number of rotations for the symmetry op
        """
        rotations = {'inf':0,
                     'XY':0,
                     'C2':2,
                     'C3':3,
                     'C4':4,
                     'C6':6,
                     'D2':2,
                     'D4':4,
                     'D6':6}
        return rotations[self.name]

    def get_n_reflections_y(self):
        """
        return number of reflections about the x axis
        """
        reflections = {'inf':0,
                       'XY':0,
                       'C2':0,
                       'C3':0,
                       'C4':0,
                       'C6':0,
                       'D2':1,
                       'D4':1,
                       'D6':1}
        return reflections[self.name]

    def get_n_reflections_xy(self):
        """
        return number of reflections about the x-y diagonal
        """
        reflections = {'inf':0,
                       'XY':1,
                       'C2':0,
                       'C3':0,
                       'C4':0,
                       'C6':0,
                       'D2':0,
                       'D4':0,
                       'D6':0}
        return reflections[self.name]

    def get_symmetry_cone_angle(self):
        """
        return the smallest angle which  covers a unique segment of the
        brillouin zone
        """
        angle = {'C2':2.*np.pi/2.0,
                 'C3':2.*np.pi/3.0,
                 'C4':2.*np.pi/4.0,
                 'C6':2.*np.pi/6.0,
                 'D2':2.*np.pi/4.0,
                 'D4':2.*np.pi/8.0,
                 'D6':2.*np.pi/12.0}
        if self.name in angle:
            return angle[self.name]
        raise ValueError("symmetry {}".format(self.name) +
                         " has no Symmetry Cone Angle")
