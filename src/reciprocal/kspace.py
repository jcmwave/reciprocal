import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from numpy.lib.scimath import sqrt as csqrt
#from kspacesampling import KVector
#from kspacesampling import BrillouinZone
#from kspacesampling import Symmetry
#import kspacesampling

class KSpace():

    """
    class obtaining 2D grids of points in reciprocal space.

    attributes:
    brillouinZone(BrillouinZone): BrillouinZone instance  which
                           defines the sampling of points inside the
                           1st Brillouin zone.
    symmetry(Symmetry): Symmetry instance defining the symmetry
                           families (see Symmetry.py for more information).
    fermi_radius(float): radius of the Fermi circle in 1/m.
    symmetry_cone(np.array<float>(3,2)): x,y vertices of the symmetry cone in 1/m.
    KSampling(np.array<float>(N,2)): all k points inside the Fermi radius.
    BF(dict): grouping of k points (Value) into Bloch families (Key).
    """

    def __init__(self, lattice):
        #self.bzone = brillouinZone
        self.lattice = lattice
        self.fermi_radius = None
        self.symmetry_cone = None
        self.bloch_families = None
        self.sampling = None
        self.symmetry_families = None

    def set_fermi_radius(self, radius):
        self.fermi_radius = radius


    def calc_woods_anomalies(self, order, phi=0.0):
        k_norm = self.fermi_radius
        #k_r = np.array([1101903.70318436]) #np.linspace(0.0, k_norm, 1000)
        #k_x_all = k_r*np.cos(np.radians(phi))
        #k_y_all = k_r*np.sin(np.radians(phi))

        vec1 = self.lattice.vectors.vec1
        vec2 = self.lattice.vectors.vec2
        woods_points = []

        # for i in range(k_r.size):
        #     kx = k_x_all[i]
        #     ky = k_y_all[i]
        #     kxy = np.array([kx, ky, 0.0])
        #     print(kxy)
        #is_woods = False
        for order1 in range(-order, order+1):
            #if is_woods:
                #break
            for order2 in range(-order, order+1):
                #if is_woods:
                #    break
                if abs(order1) < order and abs(order2) < order:                    
                    continue
                vec_total = order1*vec1 + order2*vec2

                #trial_point = (kxy + order1*vec1 + order2*vec2)/k_norm
                cos_factor = np.cos(np.radians(phi))
                sin_factor = np.sin(np.radians(phi))
                p = 2*(cos_factor*vec_total[0]+sin_factor*vec_total[1])
                q = (vec_total[0]**2+vec_total[1]**2-k_norm**2)
                for solution in [-1., 1.]:
                    k_r1 = (-0.5*p) +solution*csqrt( (0.5*p)**2 -q)
                    #k_r2 = (-0.5*p) + csqrt( (0.5*p)**2 -q)
                    if (np.abs(np.imag(k_r1)) == 0.0 and
                        0 <= np.real(k_r1) <= k_norm):
                        kx = k_r1*np.cos(np.radians(phi))
                        ky = k_r1*np.sin(np.radians(phi))
                        kxy = np.array([kx, ky])
                        already_included = False
                        for point in woods_points:
                            if (np.isclose(kxy[0], point[0]) and
                                np.isclose(kxy[1], point[1])):
                                already_included = True
                        if not already_included:
                            woods_points.append(kxy)
                            #is_woods = True

        #woods_points = np.array(woods_points)
        return woods_points






    # def plotFermiCircle(self, ax):
    #     if self.fermi_radius is None:
    #         return
    #     FC = plt.Circle((0,0),radius=self.fermi_radius,edgecolor='k',linestyle='--',linewidth=2.0,fill=False)
    #     ax.add_artist(FC)

    def calcsymmetry_cone(self):
        if self.fermi_radius is None:
            return
        angle0 = 0.0
        angle1 = self.symmetry.getsymmetry_coneAngle()
        xy1 = np.array([0.0,0.0])
        xy2 = np.array([self.fermi_radius,0.0])
        xy0 = self.fermi_radius*np.array([np.cos(angle1),np.sin(angle1)])
        self.symmetry_cone = np.array([xy0,xy1,xy2])

    def plotsymmetry_cone(self,ax):
        self.calcsymmetry_cone()
        symmetry_cone = Polygon(self.symmetry_cone,edgecolor='k',
                               fill=False,closed=False,linestyle='--',linewidth=2.0)
        ax.add_artist(symmetry_cone)

    def calcKSampling(self,nPoints=5,nPointsTotal=None,cutOffTol=1e-5,useSymmetry=True):
        if self.brillouinZone.BZ is None:
            self.brillouinZone.calcBZPolygon()
        if self.brillouinZone.IBZ is None:
            self.brillouinZone.calcIBZPolygon()
        if self.symmetry_cone is None:
            self.calcsymmetry_cone()

        if nPointsTotal is not None:
            fr = self.fermi_radius
            nPoints = self.brillouinZone.getPointsPerSideForfermi_radius(fr,nPointsTotal)

        self.brillouinZone.calcBZSampling(nPoints)
        angle0 = 0.0
        openingAngle = np.arctan2( self.symmetry_cone[0,1],self.symmetry_cone[0,0])
        Nk = self.brillouinZone.BZSampling.shape[0]
        longest_vector = 0.0
        for keyVal in self.brillouinZone.symmetryPoints:
            if np.linalg.norm(keyVal[1]) > longest_vector:
                longest_vector = np.linalg.norm(keyVal[1])
        N = int(np.ceil(self.fermi_radius/longest_vector))
        if useSymmetry:
            b1Range = range(N+1)
            b2Range = range(N+1)
        else:
            b1Range = range(-N,N+1)
            b2Range = range(-N,N+1)
        kxy = []
        blochFamilies = {}
        symmetryFamilies = {}

        b1 = self.brillouinZone.b1[:2]
        b2 = self.brillouinZone.b2[:2]
        counter = 0
        for ik in range(Nk):
            k = self.brillouinZone.BZSampling[ik]
            bFamily = []
            for nx in b1Range:
                for ny in b2Range:
                    trialPoint = nx*b1 + ny*b2 + k
                    if useSymmetry:
                        if not self.testInsidesymmetry_cone(trialPoint,openingAngle):
                            continue
                    else:
                        length = np.linalg.norm(trialPoint)
                        tol = cutOffTol
                        if length > self.fermi_radius*(1-tol):
                            continue

                    #check no longer needed as duplicates are excluded due to BZSampling construction
                    #if self.testForDuplicates(kxy,trialPoint):
                    #    continue

                    kxy.append(trialPoint)
                    bFamily.append(trialPoint)

                    if useSymmetry:
                        # construct symmetryFamily for current point
                        G = self.brillouinZone.symmetryPoints[1][1]
                        symmetry = ''
                        if kspacesampling.liesOnVertex(trialPoint,self.brillouinZone.symmetryPoints[:1])[0]:
                            #Check if trialPoint is Gamma
                            symmetry = Symmetry.Symmetry('inf')
                        elif kspacesampling.liesOnPoly(trialPoint,self.symmetry_cone,closed=False,rel_tol=1e-6):
                            symmetry = self.symmetry
                            if symmetry.name[0] == 'D':
                                symmetry = Symmetry.Symmetry('C'+symmetry.name[1:])
                        else:
                            symmetry = self.symmetry
                        points,operators = kspacesampling.applySymmetryOperators(trialPoint,symmetry)
                        sFamily = (np.array(points),operators)
                    else:
                        sFamily = (np.array(trialPoint),[np.eye(2)])

                    symmetryFamilies[counter] = sFamily
                    counter += 1

            if len(bFamily) > 0:
                famN = ik
                blochFamilies[famN] = np.array(bFamily)

        self.bloch_families = blochFamilies
        self.symmetry_families = symmetryFamilies
        self.sampling = np.array(kxy)

    def convertToKVectors(self,wavelength,n,direction):
        nRows = self.sampling.shape[0]
        n_array = np.ones(nRows)*n
        direction_array = np.ones(nRows)*direction
        self.sampling = KVector.KVectorGroup(wavelength,nRows,
                                                  kx=self.sampling[:,0],
                                                  ky=self.sampling[:,1],
                                                  n=n_array,normal=direction_array)
        for bf in self.bloch_families.keys():
            family = self.bloch_families[bf]
            nRows = family.shape[0]
            n_array = np.ones(nRows)*n
            direction_array = np.ones(nRows)*direction
            self.bloch_families[bf] = KVector.KVectorGroup(wavelength,nRows,
                                                   kx=family[:,0],
                                                   ky=family[:,1],
                                                   n=n_array,normal=direction_array)




    def testInsidesymmetry_cone(self,trialPoint,openingAngle):
        tol = 1e-3
        if trialPoint[0] < 0:
            return False
        if trialPoint[1] < -tol:
            return False
        if trialPoint[1]> trialPoint[0]*np.tan(openingAngle)+tol:
            return False
        ktrans_sq = trialPoint[0]**2 + trialPoint[1]**2
        if self.fermi_radius**2 - ktrans_sq < 0.0:
            return False
        #kz_sq = (self.n_inc*self.k_0)**2 - ktrans_sq
        return True

    def testForDuplicates(self,oldPoints,newPoint):
        already_included = False
        for kk,sampled in enumerate(oldPoints):
            if np.isclose(newPoint, sampled,rtol=1e-2,atol=1e-4).all():
                already_included = True
                if already_included:
                    return True
        return False

    @staticmethod
    def chooseColor(item,NItems):
        if NItems <= 10:
            cmap = cm.get_cmap('tab10')
            NColors =10.0
        elif NItems > 10 and NItems <=20:
            cmap = cm.get_cmap('tab20')
            NColors =20.0
        elif NItems > 20 and NItems <=40:
            cmap = cm.get_cmap('tab20c')
            NColors =40.0
        else:
            cmap = cm.get_cmap('jet')
            NColors = float(NItems)
        color = np.zeros((1,4))
        color[0,:] = cmap((float(item))/NColors)
        return color

    def plotKSampling(self,ax,n='all',color = None):
        if self.sampling is None:
            self.calcKSampling()
        plt.sca(ax)
        nPoints = self.sampling.shape[0]
        if color is None:
            if nPoints<= 10:
                cmap = cm.get_cmap('tab10')
                NColors =10.0
            elif nPoints > 10 and nPoints <=20:
                cmap = cm.get_cmap('tab20')
                NColors =20.0
            elif nPoints > 20 and nPoints <=40:
                cmap = cm.get_cmap('tab20c')
                NColors =40.0
            else:
                cmap = cm.get_cmap('jet')
                NColors =float(nPoints)
            colors = cmap((np.arange(nPoints)/NColors))
        else:
            colors=color

        if n == 'all':
            plt.scatter(self.sampling[:,0],self.sampling[:,1],c=colors)
        else:
            plt.scatter(self.sampling[:n,0],self.sampling[:n,1],c=colors)

    def plotBlochFamilies(self,ax,n='all',legend=False):
        if self.bloch_families is None:
            self.calcKSampling()
        if n=='all':
            NF = self.brillouinZone.BZSampling.shape[0]
        else:
            NF = n
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        plt.sca(ax)
        i = 0
        nFam = 0
        for key, family in sorted(self.bloch_families.items()):
            color = self.chooseColor(key,NF)
            if n != 'all' and nFam >= n:
                return
            label = "Bloch Family {}".format(key+1)
            if isinstance(family,np.ndarray):
                plt.scatter(family[:,0],family[:,1],color=color,label=label)
            else:
                plt.scatter(family.k[:,0],family.k[:,1],color=color,label=label)
            nFam += 1
        if legend:
            plt.legend(bbox_to_anchor=[1.01,0.99],loc='upper left')

    def plotSymmetryFamilies(self,ax,n='all',color=None):
        if self.symmetry_families is None:
            self.calcKSampling()

        plt.sca(ax)

        if n=='all':
            nPoints = self.sampling.shape[0]
        else:
            nPoints =n

        if color == None:
            if nPoints<= 10:
                cmap = cm.get_cmap('tab10')
                NColors =10.0
            elif nPoints > 10 and nPoints <=20:
                cmap = cm.get_cmap('tab20')
                NColors =20.0
            elif nPoints > 20 and nPoints <=40:
                cmap = cm.get_cmap('tab20c')
                NColors =40.0
            else:
                cmap = cm.get_cmap('jet')
                NColors =float(nPoints)
            colors = cmap((np.arange(nPoints)/NColors))

        if color is None:
            for i in range(nPoints):
                family = self.symmetry_families[i][0]
                color = colors[i].reshape(1,4)
                plt.scatter(family[:,0],family[:,1],c=color)
        if color == "FromBlochFamilies":
            #nBFs = np.amax(np.array(list(self.bloch_families.keys())))
            #nBFs = len(self.bloch_families.keys())
            nBFs = self.brillouinZone.BZSampling.shape[0]
            for i in range(nPoints):
                family = self.symmetry_families[i][0]
                eldest = family[0,:]
                color = 'k'
                for key,bf in sorted(self.bloch_families.items()):
                    xCheck = np.isclose( eldest[0],bf[:,0],rtol=1e-6)
                    yCheck = np.isclose( eldest[1],bf[:,1],rtol=1e-6)
                    if np.any(xCheck*yCheck):
                        color = self.chooseColor(key,nBFs)
                plt.scatter(family[:,0],family[:,1],c=color)
        else:
            for i in range(nPoints):
                family = self.symmetry_families[i][0]
                plt.scatter(family[:,0],family[:,1],c=color)



    def getBFStatistics(self):
        if self.bloch_families is None:
            return
        info = {}
        info['NFamilies'] = len(self.bloch_families)
        FamilySizes = {}
        nK = 0
        for key in self.bloch_families:
            bf = self.bloch_families[key]
            if isinstance(bf,np.ndarray):
                nSiblings = bf.shape[0]
            else:
                nSiblings = bf.k.shape[0]
            nK += nSiblings
            FamilySizes[key] = nSiblings
        info['NKPoints'] = nK
        info['FamilySizes'] = FamilySizes
        info['Speedup'] = info['NKPoints']/info['NFamilies']
        return info
