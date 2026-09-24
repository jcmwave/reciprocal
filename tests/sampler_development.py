import numpy as np
import matplotlib.pyplot as plt
from reciprocal import lattice, kspace, canvas

lat_vec_args = {}
lat_vec_args['length1'] = np.pi*2
lat_vec_args['length2'] = np.pi*2
lat_vec_args['angle'] = 90.
lat = lattice.Lattice.from_lat_vec_args(**lat_vec_args)
rlat = lat.make_reciprocal()


wvl = np.pi*1.1
k = np.pi*2/wvl
symmetry = 'D4'    
kspace_obj = kspace.KSpace(wvl, symmetry=symmetry, fermi_radius=k)
kspace_obj.apply_lattice(rlat)
kvectors = kspace_obj.periodic_sampler.sample()

fig, ax = plt.subplots(1,1, figsize=(8, 8))

can = canvas.Canvas(ax=ax)

can.plot_fermi_circle(kspace_obj)
can.plot_lattice(rlat)
can.plot_bzone(rlat.unit_cell)
lim = 2.1
plt.xlim([-lim, lim])
plt.ylim([-lim, lim])

sample = kspace_obj.periodic_sampler.sample(restrict_to_sym_cone=True)

can.plot_sampling(sample.k)