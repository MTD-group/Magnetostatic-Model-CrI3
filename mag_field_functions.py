from ase import Atoms
import numpy as np 

def cal_mag_x(mag_point, scan_point, atom_len,atom_shape):  
        ''' Calculate the x component for the magnetic field with out the prefactor for one of the spin density grids in an  
        magnetic block (consisting of tesseated spin deinsty fields) on one point of interest  
        
        Parameters 
        ------------
        mag point: ndarray 
         coordinates of the point in the magnetic block 
        scan point: (ndarray)
            coordinates of point of interest 
        atom_len ndarray 
            lengths of the sides of the spin density grid 
        atom_shape ndarray
            dimensions of the spin denisty grid 
        
        Returns
        -------
        mag : ndarray 
         contirbtuions to the x component of the magnetic field (without prefactor) at scan point
        '''
        ns=[1,2]
        nums=[0,1,2] 
        ints = [0,-1,1]
        cell_len_x = atom_len[0]/(2*atom_shape[0])
        cell_len_y = atom_len[1]/(2*atom_shape[1])
        cell_len_z= atom_len[2]/(2*atom_shape[2])
        X=scan_point[0]
        Y=scan_point[1]
        Z=scan_point[2]
        mag = 0 
        for i in ns: 
            for j in ns: 
                for k in ns: 
                    part = np.sqrt(np.add(np.add(np.square(X-(mag_point[0,:,:,:]+(ints[i]*cell_len_x))),np.square(Y-(mag_point[1,:,:,:]+(ints[j]*cell_len_y)))),np.square(Z-(mag_point[2,:,:,:]+(ints[k]*cell_len_z))))) 
                    mag_1 = -np.arctan(((Y-(mag_point[1,:,:,:]+(ints[j]*cell_len_y)))*(Z-(mag_point[2,:,:,:]+(ints[k]*cell_len_z))))/(((X-(mag_point[0,:,:,:]+(ints[i]*cell_len_x)))*part)))* np.power((-1),(nums[i]+nums[j]+nums[k]))
                    mag = mag+mag_1
                
        return(mag) 
def cal_mag_y(mag_point, scan_point, atom_len, atom_shape):       
        ''' Calculate the x component for the magnetic field with out the prefactor for one of the spin density grids in an  
        magnetic block (consisting of tesseated spin deinsty fields) on one point of interest  
        
        Parameters 
        ------------
        mag point: ndarray 
         coordinates of the point in the magnetic block 
        scan point: (ndarray)
            coordinates of point of interest 
        atom_len ndarray 
            lengths of the sides of the spin density grid 
        atom_shape ndarray
            dimensions of the spin denisty grid 
        
        Returns
        -------
        mag : ndarray 
         contirbtuions to y component of the magnetic field (without prefactor) at scan point
        '''
        ns=[1,2]
        nums=[0,1,2] 
        ints = [0,-1,1]
        cell_len_x = atom_len[0]/(2*atom_shape[0])
        cell_len_y = atom_len[1]/(2*atom_shape[1])
        cell_len_z= atom_len[2]/(2*atom_shape[2])
        X=scan_point[0]
        Y=scan_point[1]
        Z=scan_point[2]
        mag = 0 
        for i in ns: 
            for j in ns: 
                for k in ns: 
                    part = np.sqrt(np.add(np.add(np.square(X-(mag_point[0,:,:,:]+(ints[i]*cell_len_x))),np.square(Y-(mag_point[1,:,:,:]+(ints[j]*cell_len_y)))),np.square(Z-(mag_point[2,:,:,:]+(ints[k]*cell_len_z))))) 
                    mag_1 = -np.log(((Z-(mag_point[2,:,:,:]+(ints[k]*cell_len_z)))+part))* np.power((-1),(nums[i]+nums[j]+nums[k]))  
                    mag = mag+mag_1
                
        return(mag)
def cal_mag_z(mag_point, scan_point, atom_len,atom_shape): 
        ''' Calculate the x component for the magnetic field with out the prefactor for one of the spin density grids in an  
        magnetic block (consisting of tesseated spin deinsty fields) on one point of interest  
        
        Parameters 
        ------------
        mag point: ndarray 
         coordinates of the point in the magnetic block 
        scan point: (ndarray)
            coordinates of point of interest 
        atom_len ndarray 
            lengths of the sides of the spin density grid 
        atom_shape ndarray
            dimensions of the spin denisty grid 
        
        Returns
        -------
        mag : ndarray 
         contirbtuions to z component of the magnetic field (without prefactor) at scan point
        '''
        ns=[1,2]
        nums=[0,1,2] 
        ints = [0,-1,1]
        cell_len_x = atom_len[0]/(2*atom_shape[0])
        cell_len_y = atom_len[1]/(2*atom_shape[1])
        cell_len_z= atom_len[2]/(2*atom_shape[2])
        X=scan_point[0]
        Y=scan_point[1]
        Z=scan_point[2]
        mag = 0 
        for i in ns: 
            for j in ns: 
                for k in ns: 
                    part = np.sqrt(np.add(np.add(np.square(X-(mag_point[0,:,:,:]+(ints[i]*cell_len_x))),np.square(Y-(mag_point[1,:,:,:]+(ints[j]*cell_len_y)))),np.square(Z-(mag_point[2,:,:,:]+(ints[k]*cell_len_z))))) 

                    mag_1 = -np.log((Y-(mag_point[1,:,:,:]+(ints[j]*cell_len_y)))+part)* np.power((-1),(nums[i]+nums[j]+nums[k]))  
                    mag = mag+mag_1
        return(mag) 
def calc_mag_field_point(SD, mag_point, scan_point, atom_len):    
    ''' Calls functions for calcualting the magnetic field componenets then adds relevant prefactors 
    the magnetic field contibution to one point 
    
    Parameters 
    -----------
    SD: ndarray
        spin density grid 
    mag_point: ndarray 
        the point in the magnetic block where the origin of the spin density is located 
    scan_point: ndarray 
        the point in the scan grid
    atom_len: ndarray 
        dimnsion (in angstroms) of the spin density grid 
        
    Returns
    ------- 
    field: ndarray 
        x,y, and z compents of the magnetic field from mag_point at scan_point
    '''
    atom_shape = SD.shape 
    points = np.mgrid[mag_point[0]:mag_point[0]+atom_len[0]:SD.shape[0]*1j,mag_point[1]:mag_point[1]+atom_len[1]:SD.shape[1]*1j, 0:-atom_len[2]:SD.shape[2]*1j] 
    mag_x = cal_mag_x(points, scan_point,atom_len,atom_shape) 
    mag_y = cal_mag_y(points, scan_point,atom_len,atom_shape)
    mag_z =  cal_mag_z(points, scan_point,atom_len,atom_shape) 
    Mz = SD
    V_pre = Mz*9.27009994e-24*np.power(1e9,3)*1.256637e-6  
    field = [np.sum((V_pre/(4*np.pi))*mag_x),np.sum((V_pre/(4*np.pi))*mag_y),np.sum((V_pre/(4*np.pi))*+mag_z)]    
    return(field) 
def calc_mag_field_3D_ur(SD, grid_size, dz, sample_interval, mag_obj_dims, atom_len,r_num=3):  
    ''' Sets up and calcualted the magnetic field for the upper right portion of the magnetic field     
    
    Parameters 
    -----------
    SD: ndarray
        spin density grid 
    gird size: ndarray 
        size of the final mangetic field grid (in angstroms) 
    dz: 
        distance of the mangetic field grid from the magnetic block (in angstroms)
    sample_interval: 
        spacing (in micrometers) between the grid cells 
    sample_interval: 
        spacing (in micrometers) between the grid cells  
    mag_obj_dims:   
        dimensions of the magnetic block in angstroms
    atom_len:  
        dimensions of the cell corresponding to the spin density grid
    r_num: 
        number of digits to round to for the boolean opperations
        
    Returns
    ------- 
    field_x: ndarray 
        2D array of the x component of the magnetic field for the upper right portion of the field
    field_y: ndarray  
        2D array of the y component of the magnetic field for the upper right portion of the field
    field_z: ndarray 
        2D array of the z component of the magnetic field for the upper right portion of the field
    '''
    field_cumulative_x = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_cumulative_y = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_cumulative_z = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_zeros = [0,0,0]    
    grid_x= np.arange(0, grid_size[0], sample_interval[0])
    grid_y= np.arange(0,grid_size[1],sample_interval[1])
    mag_coords_x = grid_x[(grid_x >=mag_obj_dims[0][0]) & (grid_x <=mag_obj_dims[0][1])]
    mag_coords_y = grid_y[(grid_y >=mag_obj_dims[1][0]) & (grid_y <= mag_obj_dims[1][1])]
    mag_obj_dims_act=[[mag_coords_x[0],mag_coords_x[-1]],[mag_coords_y[0],mag_coords_y[-1]]]
    diags_upper_right_x = grid_x[grid_x <= mag_obj_dims_act[0][1]]  
    diags_upper_right_y = grid_y[grid_y <= mag_obj_dims_act[1][1]]
    coords=np.meshgrid(grid_x,grid_y)
    for i in diags_upper_right_x: 
        print(i)
        for j in diags_upper_right_y:  
            field = calc_mag_field_point(SD,[mag_obj_dims_act[0][1],mag_obj_dims_act[1][1],0],[i,j,dz],atom_len) 
            field_cumulative_x = field_cumulative_x+(((coords[0]<=i)&(coords[0]>=np.round((i-(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0])),r_num)))&((coords[1]<=j)&(coords[1]>=np.round(j-(mag_obj_dims_act[1][1]-mag_obj_dims_act[1][0]),r_num))))*field[0]
            field_cumulative_y = field_cumulative_y+(((coords[0]<=i)&(coords[0]>=np.round((i-(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0])),r_num)))&((coords[1]<=j)&(coords[1]>=np.round(j-(mag_obj_dims_act[1][1]-mag_obj_dims_act[1][0]),r_num))))*field[1]
            field_cumulative_z = field_cumulative_z+(((coords[0]<=i)&(coords[0]>=np.round((i-(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0])),r_num)))&((coords[1]<=j)&(coords[1]>=np.round(j-(mag_obj_dims_act[1][1]-mag_obj_dims_act[1][0]),r_num))))*field[2]

    return (field_cumulative_x,field_cumulative_y,field_cumulative_z)  
def calc_mag_field_3D_ul(SD, grid_size, dz, sample_interval, mag_obj_dims, atom_len,r_num=3): 
    ''' Sets up and calcualted the magnetic field for the upper left portion of the magnetic field     
    
    Parameters 
    -----------
    SD: ndarray
        spin density grid 
    gird size: ndarray 
        size of the final mangetic field grid (in angstroms) 
    dz: 
        distance of the mangetic field grid from the magnetic block (in angstroms)
    sample_interval: 
        spacing (in micrometers) between the grid cells 
    sample_interval: 
        spacing (in micrometers) between the grid cells  
    mag_obj_dims:   
        dimensions of the magnetic block in angstroms
    atom_len:  
        dimensions of the cell corresponding to the spin density grid
    r_num: 
        number of digits to round to for the boolean opperations
        
    Returns
    ------- 
    field_x: ndarray 
        2D array of the x component of the magnetic field for the upper left portion of the field
    field_y: ndarray  
        2D array of the y component of the magnetic field for the upper left portion of the field
    field_z: ndarray 
        2D array of the z component of the magnetic field for the upper left portion of the field
    '''
    field_cumulative_x = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_cumulative_y = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_cumulative_z = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_zeros = [0,0,0]    
    grid_x= np.arange(0, grid_size[0], sample_interval[0])
    grid_y= np.arange(0,grid_size[1],sample_interval[1])
    mag_coords_x = grid_x[(grid_x >=mag_obj_dims[0][0]) & (grid_x <=mag_obj_dims[0][1])]
    mag_coords_y = grid_y[(grid_y >=mag_obj_dims[1][0]) & (grid_y <= mag_obj_dims[1][1])]
    mag_obj_dims_act=[[mag_coords_x[0],mag_coords_x[-1]],[mag_coords_y[0],mag_coords_y[-1]]]

    diags_upper_left_x = grid_x[grid_x>=mag_obj_dims_act[0][0]+sample_interval[0]]
    diags_upper_left_y = grid_y[grid_y<=mag_obj_dims_act[1][1]-sample_interval[1]]
    coords=np.meshgrid(grid_x,grid_y)
    for i in diags_upper_left_x: 
        print(i)
        for j in diags_upper_left_y:  
            #print(j)
            field = calc_mag_field_point(SD,[mag_obj_dims_act[0][0],mag_obj_dims_act[1][1],0],[i,j,dz],atom_len) 
            field_cumulative_x = field_cumulative_x+(((coords[0]>=i)&(coords[0]<np.round((i+(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0])),r_num)))&((coords[1]<=j)&(coords[1]>np.round((j-(mag_obj_dims_act[1][1]-mag_obj_dims_act[1][0])),r_num))))*field[0]
            field_cumulative_y = field_cumulative_y+(((coords[0]>=i)&(coords[0]<np.round((i+(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0])),r_num)))&((coords[1]<=j)&(coords[1]>np.round((j-(mag_obj_dims_act[1][1]-mag_obj_dims_act[1][0])),r_num))))*field[1]
            field_cumulative_z = field_cumulative_z+(((coords[0]>=i)&(coords[0]<np.round((i+(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0])),r_num)))&((coords[1]<=j)&(coords[1]>np.round((j-(mag_obj_dims_act[1][1]-mag_obj_dims_act[1][0])),r_num))))*field[2]   
    return (field_cumulative_x,field_cumulative_y,field_cumulative_z)  
def calc_mag_field_3D_lr(SD, grid_size, dz, sample_interval, mag_obj_dims, atom_len,r_num=3): 
    ''' Sets up and calcualted the magnetic field for the lower right portion of the magnetic field     
    
    Parameters 
    -----------
    SD: ndarray
        spin density grid 
    gird size: ndarray 
        size of the final mangetic field grid (in angstroms) 
    dz: 
        distance of the mangetic field grid from the magnetic block (in angstroms)
    sample_interval: 
        spacing (in micrometers) between the grid cells 
    sample_interval: 
        spacing (in micrometers) between the grid cells  
    mag_obj_dims:   
        dimensions of the magnetic block in angstroms
    atom_len:  
        dimensions of the cell corresponding to the spin density grid
    r_num: 
        number of digits to round to for the boolean opperations
        
    Returns
    ------- 
    field_x: ndarray 
        2D array of the x component of the magnetic field for the lower right portion of the field
    field_y: ndarray  
        2D array of the y component of the magnetic field for the lower right portion of the field
    field_z: ndarray 
        2D array of the z component of the magnetic field for the lower right portion of the field
    '''
    field_cumulative_x = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_cumulative_y = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_cumulative_z = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_zeros = [0,0,0]   
    grid_x= np.arange(0, grid_size[0], sample_interval[0])
    grid_y= np.arange(0,grid_size[1], sample_interval[1])
    mag_coords_x = grid_x[(grid_x >=mag_obj_dims[0][0]) & (grid_x <=mag_obj_dims[0][1])]
    mag_coords_y = grid_y[(grid_y >=mag_obj_dims[1][0]) & (grid_y <= mag_obj_dims[1][1])]
    mag_obj_dims_act=[[mag_coords_x[0],mag_coords_x[-1]],[mag_coords_y[0],mag_coords_y[-1]]]
    diags_lower_right_x = grid_x[grid_x <=mag_obj_dims[0][1]-sample_interval[0]]
    diags_lower_right_y = grid_y[grid_y >=mag_obj_dims[1][0]+sample_interval[1]]
    coords=np.meshgrid(grid_x,grid_y)
    for i in diags_lower_right_x: 
        print(i)
        for j in diags_lower_right_y:  
            field = calc_mag_field_point(SD,[mag_obj_dims_act[0][1],mag_obj_dims_act[1][0],0],[i,j,dz],atom_len) 
            field_cumulative_x = field_cumulative_x+(((coords[0]<=i)&(coords[0]>np.round(i-(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0]),r_num)))&((coords[1]>=j)&(coords[1]<np.round(j+(mag_obj_dims_act[1][1]-mag_obj_dims_act[1][0]),r_num))))*field[0]
            field_cumulative_y = field_cumulative_y+(((coords[0]<=i)&(coords[0]>np.round(i-(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0]),r_num)))&((coords[1]>=j)&(coords[1]<np.round(j+(mag_obj_dims_act[1][1]-mag_obj_dims_act[1][0]),r_num))))*field[1]
            field_cumulative_z = field_cumulative_z+(((coords[0]<=i)&(coords[0]>np.round(i-(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0]),r_num)))&((coords[1]>=j)&(coords[1]<np.round(j+(mag_obj_dims_act[1][1]-mag_obj_dims_act[1][0]),r_num))))*field[2]   
    
    return (field_cumulative_x,field_cumulative_y,field_cumulative_z)  
def calc_mag_field_3D_ll(SD, grid_size, dz, sample_interval, mag_obj_dims, atom_len,r_num=3): 
    ''' Sets up and calcualted the magnetic field for the lower left portion of the magnetic field     
    
    Parameters 
    -----------
    SD: ndarray
        spin density grid 
    gird size: ndarray 
        size of the final mangetic field grid (in angstroms) 
    dz: 
        distance of the mangetic field grid from the magnetic block (in angstroms)
    sample_interval: 
        spacing (in micrometers) between the grid cells 
    sample_interval: 
        spacing (in micrometers) between the grid cells  
    mag_obj_dims:   
        dimensions of the magnetic block in angstroms
    atom_len:  
        dimensions of the cell corresponding to the spin density grid
    r_num: 
        number of digits to round to for the boolean opperations
        
    Returns
    ------- 
    field_x: ndarray 
        2D array of the x component of the magnetic field for the lower left portion of the field
    field_y: ndarray  
        2D array of the y component of the magnetic field for the lower left portion of the field
    field_z: ndarray 
        2D array of the z component of the magnetic field for the lower left portion of the field
    '''
    field_cumulative_x = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_cumulative_y = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_cumulative_z = np.zeros([int(grid_size[1]/sample_interval[1]),int(grid_size[0]/sample_interval[0])]) 
    field_zeros = [0,0,0]
    grid_x= np.arange(0, grid_size[0], sample_interval[0])
    grid_y= np.arange(0,grid_size[1],sample_interval[1])
    mag_coords_x = grid_x[(grid_x >=mag_obj_dims[0][0]) & (grid_x <=mag_obj_dims[0][1])]
    mag_coords_y = grid_y[(grid_y >=mag_obj_dims[1][0]) & (grid_y <= mag_obj_dims[1][1])]
    mag_obj_dims_act=[[mag_coords_x[0],mag_coords_x[-1]],[mag_coords_y[0],mag_coords_y[-1]]]
    diags_lower_left_x = grid_x[grid_x >= mag_obj_dims[0][0]] 
    diags_lower_left_y = grid_y[grid_y>= mag_obj_dims[1][0]]
    coords=np.meshgrid(grid_x,grid_y) 
    for i in diags_lower_left_x: 
        print(i)
        for j in diags_lower_left_y:   

            field = calc_mag_field_point(SD,[mag_obj_dims_act[0][0],mag_obj_dims_act[1][0],0],[i,j,dz],atom_len) 
            field_cumulative_x = field_cumulative_x+(((coords[0]>=i)&(coords[0]<np.round(i+(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0]),r_num)))&((coords[1]>=j)&(coords[1]<np.round(j+(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0]),r_num))))*field[0]
            field_cumulative_y = field_cumulative_y+(((coords[0]>=i)&(coords[0]<np.round(i+(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0]),r_num)))&((coords[1]>=j)&(coords[1]<np.round(j+(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0]),r_num))))*field[1]
            field_cumulative_z = field_cumulative_z+(((coords[0]>=i)&(coords[0]<np.round(i+(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0]),r_num)))&((coords[1]>=j)&(coords[1]<np.round(j+(mag_obj_dims_act[0][1]-mag_obj_dims_act[0][0]),r_num))))*field[2]
    
    
    return (field_cumulative_x,field_cumulative_y,field_cumulative_z)     
    