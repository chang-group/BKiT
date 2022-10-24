import sys
import os
import pytraj as pt
import numpy as np
import tqdm
from ConstructMilestones3D import rotation_matrix

# for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d

def SplitEvenOdd(N):
    """
        
    """    
    if N%2 == 0:
        N = N - 1
        
    ids = np.arange(N)
    even = ids[ids%2==0]
    odd = ids[ids%2!=0]
     
    return even, odd

def LoadTrajs(traj_path, top_path, refPDB_path, mask_selec, mask_align=None):
    """ 
    
    """
    
    traj = pt.iterload(traj_path, top_path, stride=1)
    n_res = traj.top.n_residues
    nframe = traj.n_frames
    n_atoms = traj.n_atoms
    
    print('Total number of residues -- ' , n_res)
    print('Total number of atoms -- ' , n_atoms)
    print('Total number of frames -- ' , nframe)
    
    maskS1 = mask_selec[0]#+",:" + str(n_res)
    maskS2 = mask_selec[1] 
   
    if mask_align == None:
        mask_align = "(:1-" + str(n_res-1) + ")&(@N,@C,@CA,@O)"
        print('Using entire sequence backbone to superimpose all frames')
        
    refpdb = pt.load(refPDB_path) 
    pt.superpose(traj, ref=refpdb, mask=mask_align)
    #traj._force_load = True # importand when data exceeds ram
    
    chunksize = min([200, nframe])
    nchunk = int(np.floor(nframe/chunksize)) + 1 if nframe%chunksize != 0 else int(np.floor(nframe/chunksize))
    pile = [] # consider np.array!

    print('Loading metadynamics trajectory by chunk... ')
    for i in tqdm.tqdm(range(nchunk)):
        tmp = traj[i*chunksize:min((i+1)*chunksize, nframe)][maskS1][maskS2]
        pile.append(tmp.xyz.reshape(tmp.n_frames, tmp.n_atoms*3))
   
    proteinUNK_2d = np.concatenate(pile,axis=0)

    # set the first frame of the trajectory as an origin
    refframe =  proteinUNK_2d[0,:].copy()
    new_frames = proteinUNK_2d - refframe
    print('Total number of selected atoms -- ' , new_frames.shape[1]//3)

    return new_frames, refframe, traj


def pathpatch_2d_to_3d(pathpatch, delta, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    
    Borrowed from stack overflow
    
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1,0,0), index)

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D
    M = rotation_matrix(normal, (0, 0, 1)) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) for x, y in verts])
    pathpatch._segment3d += delta



def Generate2Data(n, x_max=12):
    """
    
    """
    
    rand = np.random.rand(n)*2 - 1
    
    x1_max = x_max/5
    x1 = np.arange(0, x1_max , x1_max/n)
    y1 = 1.0*np.sin(.4*x1) + .2*rand

    x2_max = x1_max + 0.6*x_max
    x2 = np.arange(x1_max, x2_max, 0.6*x_max/n)
    y2 = 1.0*np.sin(.4*x2 ) + .4*rand
    
    x3 = np.arange(x2_max, x_max, 0.2*x_max/n)
    y3 = 1.0*np.cos(.4*x3 ) + .3*rand
    
    dat = np.concatenate([np.column_stack([x1,y1]),
                          np.column_stack([x2,y2]),
                          np.column_stack([x3,y3])])
    
    return dat
    

def Generate3Data(n, x_max=20.):
    """
    
    """
    
    omega = 0.3
    noiseA = 1.6
    randY = (np.random.rand(n)*2 - 1) * noiseA
    randZ = (np.random.rand(n)*2 - 1) * noiseA

    x1_max = 0.2*x_max
    x1 = np.arange(0, x1_max , x1_max/n) 
    y1 = 1.5*np.sin(omega*x1) + randY
    z1 = 1.5*np.cos(omega*x1) + randZ

    x2_max = x1_max + 0.6*x_max
    x2 = np.arange(x1_max, x2_max, 0.6*x_max/n) 
    y2 = np.sin(omega*x2 ) + randY
    z2 = np.cos(omega*x2 ) + randZ

    
    x3 = np.arange(x2_max, x_max, 0.2*x_max/n) 
    y3 = 2.5*np.sin(omega*x3 ) + randY
    z3 = 2.5*np.cos(omega*x3 ) + randZ

    
    dat = np.concatenate([np.column_stack([x1,y1,z1]),
                          np.column_stack([x2,y2,z2]),
                          np.column_stack([x3,y3,z3])])
    
    return dat


def PlotSelected(dat_sel, dr, pathP, normals, yz_pad, figsize=[8,6], plotOrig=False):
    """
    
    """
    n_disks = normals.shape[0]
    fig = plt.figure(figsize = figsize)
    ax = plt.axes(projection='3d')
    
   
    #plot path poits
    ax.plot3D(pathP[:,0], pathP[:,1], pathP[:,2], c='black', marker='o', markersize=2, alpha=0.8)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3') 

    #plotting disks
    for i in range(n_disks):
        c = Circle((0,0), dr, facecolor='grey', alpha=0.4)
        ax.add_patch(c)
        pathpatch_2d_to_3d(c, pathP[i], normal = normals[i])
        ax.text(pathP[i,0], pathP[i,1] + dr, pathP[i,2] + 2*dr, str(i), 'x')
    

    if plotOrig:
        p = ax.scatter3D(dat_sel[:,0], dat_sel[:,1], dat_sel[:,2], c = range(int(len(dat_sel))), alpha=0.1)
    else:
        #plot selected points
        p = ax.scatter3D(dat_sel[:,0], dat_sel[:,1], dat_sel[:,2], c = dat_sel[:,3],
                         alpha = 0.5, s = 2.8, cmap = 'prism')

    x_min, y_min, z_min = pathP.min(axis=0) 
    x_max, y_max, z_max = pathP.max(axis=0)
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min - yz_pad, y_max + yz_pad)
    ax.set_zlim3d(z_min - yz_pad, z_max + yz_pad)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    #ax.xaxis.pane.set_edgecolor('w')
    #ax.yaxis.pane.set_edgecolor('w')
    #ax.zaxis.pane.set_edgecolor('w')
    fig.tight_layout()
