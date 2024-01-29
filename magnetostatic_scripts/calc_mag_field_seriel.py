import numpy as np
from matplotlib import pyplot as plt 
import mag_field_functions  

slab = Atoms(read('CONTCAR'))
cell_dims = [slab.get_cell()[0][0],slab.get_cell()[1][1], slab.get_cell()[2][2]]
slab.get_atomic_numbers()
z_max_atom = np.argmax(slab.get_positions()[:,2])
z_min_atom = np.argmin(slab.get_positions()[:,2])
z_min = slab.get_positions()[z_min_atom,2]-covalent_radii[slab.get_atomic_numbers()[z_min_atom]]
z_max = slab.get_positions()[z_max_atom,2]+covalent_radii[slab.get_atomic_numbers()[z_max_atom]]
#print('read charge denisty')
vchg =  VaspChargeDensity(filename =  'CHGCAR' )
spin_dens=(vchg.chgdiff[0])
cell_dims[0] = cell_dims[0]*tile_factor[0]
cell_dims[1] = cell_dims[1]*tile_factor[1]
z_inc = cell_dims[2]/spin_dens.shape[2]
SD_lower_bound= int(spin_dens.shape[2]/cell_dims[2]*z_min)
SD_upper_bound= int(spin_dens.shape[2]/cell_dims[2]*z_max)
spin_dens_crop = spin_dens[:,:,SD_lower_bound:SD_upper_bound]*1000
cell_dims_crop = cell_dims.copy()
cell_dims_crop[2] = z_inc *spin_dens_crop.shape[2]
mag_obj_dims=[[0,1],[0,1]]
sample_interval = np.divide(cell_dims_crop,10000)
atom_len = np.divide(cell_dims_crop,10000)
grid_size = [int(1.5/np.divide(cell_dims_crop,10000)[0])*np.divide(cell_dims_crop,10000)[0],int(1.5/np.divide(cell_dims_crop,10000)[1])*np.divide(cell_dims_crop,10000)[1]] 
field_combined_ul = (field_x_ul+field_y_ul+field_z_ul)/np.sqrt(3)
np.save('field_x_ul.npy', field_x_ul)
np.save('field_y_ul.npy', field_y_ul)
np.save('field_z_ul.npy', field_z_ul) 
np.save('field_combined_ul.npy', field_combined_ul)

field_x_ll,field_y_ll,field_z_ll = mag_field_functions.calc_mag_field_3D_ll(spin_dens, grid_size, 0.062, np.divide(cell_dims,10000), mag_obj_dims, np.divide(cell_dims,10000),3) 

field_combined_ll = (field_x_ll+field_y_ll+field_z_ll)/np.sqrt(3)
np.save('field_x_1l.npy', field_x_ll)
np.save('field_y_1l.npy', field_y_ll)
np.save('field_z_1l.npy', field_z_ll) 
np.save('field_combined_1l.npy', field_combined_ll)




field_x_ur,field_y_ur,field_z_ur = mag_field_functions.calc_mag_field_3D_ur(spin_dens, grid_size, 0.062, np.divide(cell_dims,10000), mag_obj_dims, np.divide(cell_dims,10000),3) 

field_combined_ur = (field_x_ur+field_y_ur+field_z_ur)/np.sqrt(3)
np.save('field_x_ur.npy', field_x_ur)
np.save('field_y_ur.npy', field_y_ur)
np.save('field_z_ur.npy', field_z_ur) 
np.save('field_combined_ur.npy', field_combined_ur)

field_x_lr,field_y_lr,field_z_lr = mag_field_functions.calc_mag_field_3D_lr(spin_dens, grid_size, 0.062, np.divide(cell_dims,10000), mag_obj_dims, np.divide(cell_dims,10000),3) 

field_combined_lr = (field_x_lr+field_y_lr+field_z_lr)/np.sqrt(3)
np.save('field_x_lr.npy', field_x_lr)
np.save('field_y_lr.npy', field_y_lr)
np.save('field_z_lr.npy', field_z_lr) 
np.save('field_combined_lr.npy', field_combined_lr)

field_combined = field_combined_ul+field_combined_ll+field_combined_ur+field_combined_lr
plt.imshow(field_combined)  
plt.colorbar() 
plt.savefig('Field_combined.png') 
plt.show()
