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
grid_x= np.arange(0, grid_size[0], atom_len[0])
sub_grid_x = grid_x[grid_x >= mag_obj_dims[0][0]]
field_x = np.zeros([int(grid_size[1]/np.divide(cell_dims_crop,10000)[1]), int(grid_size[0]/np.divide(cell_dims_crop,10000)[0])])
field_y = np.zeros([int(grid_size[1]/np.divide(cell_dims_crop,10000)[1]), int(grid_size[0]/np.divide(cell_dims_crop,10000)[0])])
field_z = np.zeros([int(grid_size[1]/np.divide(cell_dims_crop,10000)[1]), int(grid_size[0]/np.divide(cell_dims_crop,10000)[0])])

if rank == 0:
    # determine the size of each sub-task
    nsteps = len(sub_grid_x)

    ave, res = divmod(nsteps, nprocs)
    counts = [ave + 1 if p < res else ave for p in range(nprocs)]
    print('counts',len(counts),counts)
    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(nprocs)]
    ends = [sum(counts[:p+1]) for p in range(nprocs)]
    print('starts',len(starts),starts)
    print('ends',len(ends),ends)
    # save the starting and ending indices in data
    data = [(starts[p], ends[p]) for p in range(nprocs)]
else:
    data = None

data = comm.scatter(data,root=0)
for i in sub_grid_x[data[0]:data[1]]:
    #print(i)
    field_x_int,field_y_int,field_z_int = calc_mag_field_3D_ll(spin_dens_crop, grid_size, 0.0015, np.divide(cell_dims_crop,10000), mag_obj_dims, np.divide(cell_dims_crop,10000),3,i)
    field_x = np.add(field_x, field_x_int)
    field_y = np.add(field_y,field_y_int)
    field_z = np.add(field_z, field_z_int)
np.save('CrI3_field_x_ll_'+str(rank)+'.npy', field_x)
np.save('CrI3_field_y_ll_'+str(rank)+'.npy', field_y)
np.save('CrI3_field_z_ll_'+str(rank)+'.npy', field_z)

if rank==0:
    # Step 3: Don't forget to close

    print('done')