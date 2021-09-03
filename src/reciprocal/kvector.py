import numpy as np
import scipy.constants as constants
class KVector(object):

    """
    class for defining a k vector in terms of both cartesian and polar
    coordinates
    """


    def __init__(self, wavelength,
                 n=None, theta=None, phi=None, normal=None,
                 kx=None, ky=None, kz=None, validate=True):
        self.wavelength = wavelength #scalar
        self.k0 = 2*np.pi/(self.wavelength)
        self.n = n
        self.theta = theta
        self.phi = phi
        self.normal_ = normal
        self.kx = kx
        self.ky = ky
        self.kz = kz
        if validate:
            combination = self.validateData()
            self.completeData(combination)

    def __repr__(self):
        return "k:{},theta:{},phi:{},normal:{},n:{}".format(self.k,self.theta,self.phi,self.normal,self.n)

    def validateData(self):
        validCombinations = []
        validCombinations.append([self.kx,self.ky,self.kz])
        validCombinations.append([self.kx,self.ky,self.n,self.normal_])
        validCombinations.append([self.theta,self.phi,self.n,self.normal_])
        valid = False
        for i in range(len(validCombinations)):
            thisCombo = True
            for j in range(len(validCombinations[i])):
                thisCombo *= validCombinations[i][j] is not None
            if thisCombo:
                valid = True
                combination = i
                break
        if valid == False:
            raise ValueError("not enough information to uniquely determine plane wave")
        return combination

    def completeData(self,combination):
        if combination == 0:
            "all 3 k components are given"
            self.n = self.getNFromK()
            self.normal_ = self.getNormalFromK()
            thetaPhi = self.getThetaPhiFromK()
            self.theta = thetaPhi[0]
            self.phi = thetaPhi[1]
        if combination == 1:
            "kx,ky plus n and normal"
            self.kz = self.getKZFromKXYNNormal()
            thetaPhi = self.getThetaPhiFromK()
            self.theta = thetaPhi[0]
            self.phi = thetaPhi[1]
        if combination == 2:
            "theta,phi plus n and normal"
            kxyz = self.getKFromThetaPhiNNormal()
            self.kx = kxyz[0]
            self.ky = kxyz[1]
            self.kz = kxyz[2]


    def getNFromK(self):
        return np.linalg.norm(self.k)/self.k0

    def getNormalFromK(self):
        return np.sign(self.kz)

    def getKZFromKXYNNormal(self):
        return self.normal_*np.sqrt(np.clip( self.knorm**2 -self.kx**2 -self.ky**2,
                                      a_min=(self.knorm**2)*1e-6,a_max=None))

    def getThetaPhiFromK(self):
        kxy = np.sqrt( self.kx**2 + self.ky**2)
        theta = np.arctan2(kxy,np.abs(self.kz))
        theta = np.degrees(theta)
        phi = np.degrees(np.arctan2(self.ky,self.kx))
        return [theta,phi]

    def getKFromThetaPhiNNormal(self):
        theta = self.theta
        phi = self.phi
        kx = self.knorm*np.cos(np.radians(phi))*np.sin(np.radians(theta))
        ky = self.knorm*np.sin(np.radians(phi))*np.sin(np.radians(theta))
        kz = self.normal_*self.knorm*np.cos(np.radians(theta))
        return [kx,ky,kz]

    @property
    def knorm(self):
        k = self.k0
        return self.n*k

    @property
    def k(self):
        return np.array([self.kx,self.ky,self.kz])

    @property
    def normal(self):
        if np.sign(self.normal_) == 1:
            return "+z"
        else:
            return "-z"


    def str(self):
        return "k:{},theta:{},phi:{},normal:{},n:{}".format(self.k,self.theta,self.phi,self.normal,self.n)

from enum import Enum

class KVectorGroupColumns(Enum):
    kx = 0
    ky = 1
    kz = 2
    theta = 3
    phi = 4
    normal = 5
    n = 6

class KVectorGroup(object):

    """
    class for defining a group of k vectors, allowing their components to be
    stored in a single array
    """

    def __init__(self, wavelength, n_rows,
                 n=None, theta=None, phi=None, normal=None,
                 kx=None, ky=None, kz=None, validate=True, data=None):
        self.wavelength = wavelength #scalar
        self.k0 = 2*np.pi/(self.wavelength)
        self.n_rows = n_rows
        self.data_ = np.empty((n_rows,7),dtype=np.float64)
        if data is None:
            self.data_.fill(float('nan'))
        else:
            self.data_ = data
        self.cols = KVectorGroupColumns
        col = 0
        for item in [kx,ky,kz,theta,phi,normal,n]:
            if item is not None:
                self.data_[:,col] = item
            col += 1
        if validate:
            combination = self.validateData()
            self.completeData(combination)

    def __repr__(self):
        return "k:{},theta:{},phi:{},normal:{},n:{}".format(self.k,self.theta,self.phi,self.normal,self.n)

    def validateData(self):
        validColumns = []
        validColumns.append([self.cols.kx,self.cols.ky,self.cols.kz])
        validColumns.append([self.cols.kx,self.cols.ky,
                             self.cols.n,self.cols.normal])
        validColumns.append([self.cols.theta,self.cols.phi,
                             self.cols.n,self.cols.normal])
        valid = False
        for i in range(len(validColumns)):
            thisCombo = True
            for j in range(len(validColumns[i])):
                col = validColumns[i][j].value
                thisCombo *= np.any(np.isnan(self.data_[:,col])) == False
            if thisCombo:
                valid = True
                combination = i
                break
        if valid == False:
            raise ValueError("not enough information to uniquely determine plane wave")
        return combination

    def completeData(self,combination):
        if combination == 0:
            "all 3 k components are given"
            self.data_[:,self.cols.n.value] = self.getNFromK()
            self.data_[:,self.cols.normal.value] = self.getNormalFromK()
            thetaPhi = self.getThetaPhiFromK()
            self.data_[:,self.cols.theta.value] = thetaPhi[0]
            self.data_[:,self.cols.phi.value] = thetaPhi[1]
        if combination == 1:
            "kx,ky plus n and normal"
            self.data_[:,self.cols.kz.value] = self.getKZFromKXYNNormal()
            thetaPhi = self.getThetaPhiFromK()
            self.data_[:,self.cols.theta.value] = thetaPhi[0]
            self.data_[:,self.cols.phi.value] = thetaPhi[1]
        if combination == 2:
            "theta,phi plus n and normal"
            kxyz = self.getKFromThetaPhiNNormal()
            self.data_[:,self.cols.kx.value] = kxyz[0]
            self.data_[:,self.cols.ky.value] = kxyz[1]
            self.data_[:,self.cols.kz.value] = kxyz[2]


    def getNFromK(self):
        return np.linalg.norm(self.k,axis=1)/self.k0

    def getNormalFromK(self):
        return np.sign(self.kz)

    def getKZFromKXYNNormal(self):
        return self.normal_*np.sqrt(np.clip( np.power(self.knorm,2)
                                            -np.power(self.kx,2)
                                            -np.power(self.ky,2),
                                            a_min=(np.power(self.knorm,2))*1e-6,
                                            a_max=None))

    def getThetaPhiFromK(self):
        kxy = np.sqrt( np.power(self.kx,2) + np.power(self.ky,2))
        theta = np.arctan2(kxy,np.abs(self.kz))
        theta = np.degrees(theta)
        phi = np.degrees(np.arctan2(self.ky,self.kx))
        return [theta,phi]

    def getKFromThetaPhiNNormal(self):
        theta = self.theta
        phi = self.phi
        kx = self.knorm*np.cos(np.radians(phi))*np.sin(np.radians(theta))
        ky = self.knorm*np.sin(np.radians(phi))*np.sin(np.radians(theta))
        kz = self.normal_*self.knorm*np.cos(np.radians(theta))
        return [kx,ky,kz]

    @property
    def knorm(self):
        k = self.k0
        return self.n*k

    @property
    def k(self):
        return self.data_[:,:3]

    @property
    def kx(self):
        return self.data_[:,self.cols.kx.value]

    @property
    def ky(self):
        return self.data_[:,self.cols.ky.value]

    @property
    def kz(self):
        return self.data_[:,self.cols.kz.value]

    @property
    def theta(self):
        return self.data_[:,self.cols.theta.value]

    @property
    def phi(self):
        return self.data_[:,self.cols.phi.value]

    @property
    def normal_(self):
        return self.data_[:,self.cols.normal.value]

    @property
    def n(self):
        return self.data_[:,self.cols.n.value]


    @property
    def normal(self):
        norm = np.empty(self.n_rows,dtype=np.dtype("U2"))
        norm[self.normal_ == 1] = "+z"
        norm[self.normal_ == -1] = "-z"
        return norm


    def sort(self,column,order='ascending'):
        if order == 'ascending':
            indices = np.argsort( -self.data_[:,self.cols[column].value])
        elif order == 'descending':
            indices = np.argsort( -self.data_[:,self.cols[column].value])
        elif order == 'absolute_ascending':
            indices = np.argsort( np.abs(self.data_[:,self.cols[column].value]))
        elif order == 'absolute_descending':
            indices = np.argsort( -np.abs(self.data_[:,self.cols[column].value]))
        self.data_ = self.data_[indices,:]


    def slice(self,row):
        datarow = self.data_[row,:]
        return KVector(self.wavelength,n=datarow[self.cols.n.value],
                         theta=datarow[self.cols.theta.value],
                         phi =datarow[self.cols.phi.value],
                         normal=datarow[self.cols.normal.value],
                         kx = datarow[self.cols.kx.value],
                         ky = datarow[self.cols.ky.value],
                         kz = datarow[self.cols.kz.value],
                         validate=False)

    def __add__(self,other):
        assert np.isclose(other.wavelength,self.wavelength)
        newNRows = self.n_rows + other.n_rows
        newData = np.concatenate( (self.data_,other.data_),axis=0)
        return KVectorGroup(self.wavelength,newNRows,
                                 data=newData,validate=False)



class BlochFamilyColumns(Enum):
    order1 = 7
    order2 = 8

class BlochFamily(KVectorGroup):

    """
    Extends KVectorGroup to include a lattice index
    """
    def __init__(self, *args, order1=None, order2=None, **kwargs):
         """Initialize a BlochFamily object

         For more information on arguemnts, see KVectorGroup

         Parameters
         ----------
         order1: (N,)<np.int>np.array
            the first lattice order
         second1: (N,)<np.int>np.array
            the second lattice order
         """
         #order1 = kwargs.pop("order1")
         #order2 = kwargs.pop("order2")
         super(BlochFamily, self).__init__(*args, **kwargs)

         extended_data = np.empty((self.n_rows,9), dtype=np.float64)
         extended_data.fill(float('nan'))
         extended_data[:,:BlochFamilyColumns.order1.value] = self.data_
         self.data_ = extended_data
         if order1 is not None:
             self.data_[:, BlochFamilyColumns.order1.value] = order1
         if order2 is not None:
             self.data_[:, BlochFamilyColumns.order2.value] = order2

    @classmethod
    def from_kvector_group(bloch_family, kvector_group):
        """Return BlochFamily from a KVectorGroup


        """
        return BlochFamily(kvector_group.wavelength,
                           kvector_group.n_rows,
                           data = kvector_group.data_,
                           validate=False)

    def set_orders(self, order1, order2):
        self.data_[:, BlochFamilyColumns.order1.value] = order1
        self.data_[:, BlochFamilyColumns.order2.value] = order2

    @property
    def order1(self):
        return self.data_[:, BlochFamilyColumns.order1.value]

    @property
    def order2(self):
        return self.data_[:, BlochFamilyColumns.order2.value]

    def __repr__(self):
        return_str =  "k:{},theta:{},phi:{},".format(self.k,self.theta,self.phi)
        return_str += "normal:{},n:{},".format(self.normal,self.n)
        return_str += "order1:{},order2:{}".format(self.order1, self.order2)
        return return_str


if __name__ == '__main__':
    pass
