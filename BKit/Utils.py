import sys
import os
import pytraj as pt
import numpy as np
import tqdm
#from ConstructMilestones3D import rotation_matrix, pathpatch_2d_to_3d

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


def PlotSelected(dat_sel, dr, pathP, normals, yz_pad, figsize=[8,6]):
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