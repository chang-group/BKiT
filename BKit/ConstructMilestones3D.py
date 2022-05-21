import sys
import os
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

def rotation_matrix(v1,v2):
    """
    Calculates the rotation matrix that changes v1 into v2.
    Borrowed from stack overflow
    """
    v1/=np.linalg.norm(v1)
    v2/=np.linalg.norm(v2)

    cos_angle=np.dot(v1,v2)
    d=np.cross(v1,v2)
    sin_angle=np.linalg.norm(d)

    if sin_angle == 0:
        M = np.identity(3) if cos_angle>0. else -np.identity(3)
    else:
        d/=sin_angle

        eye = np.eye(3)
        ddt = np.outer(d, d)
        skew = np.array([[ 0   ,  d[2], -d[1]],
                         [-d[2],  0   ,  d[0]],
                         [ d[1], -d[0],  0  ]], dtype=np.float64)

        M = ddt + cos_angle * (eye - ddt) + sin_angle * skew

    return M

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
    

class ConstructMilestones3D:
   
    def __init__(self, path):
        """
        
        """
        self.path = path
        #self.ml_length= ml_length
        
    
    def GetVectors(self):
        """
        Simply get vector components
        """
        fp = self.path
        n_rows, n_cols = fp.shape[0], fp.shape[1]
        right = fp[1:n_rows]
        left = fp[0:n_rows-1]
        dr = right - left
        dr = np.vstack((dr,dr[-1]))     
      
        return dr
    
    def GetAngles(self):
        """
        Simply get angles (theta, phi -- spherical coordinate sys)
        """
        dr = self.GetVectors()
        #print(dr[0:6])
        r = np.sqrt((dr*dr).sum(axis=1))
        angs = np.zeros(shape=(dr.shape[0],2))
        angs[:,0] = np.arccos(dr[:,2] / r)
        angs[:,1] = np.arcsin( dr[:,1] / np.sqrt(dr[:,0]*dr[:,0] + dr[:,1]*dr[:,1]))

        return angs

    
    def OptAngles(self, angles, n_iter=100, lr=0.01):
        """
        Angles are optimized by taking average over neirest
        neihbours - 'Mean field approximation'. Vectorized version! 
        
        """
        alpha = np.pad(angles, ((1, 1),(0, 0)), 'edge')
        m = alpha.shape[0] # paded n_rows
        
        for iter in range(n_iter): 
            alpha_mean = 0.5*(alpha[0:m-2] + alpha[2:m])
            dalpha = -alpha[1:m-1] + alpha_mean 
            alpha[1:m-1] += lr*dalpha
                
        return  alpha[1:m-1] # removed pads
    
    
    def OptVectors(self, vecs, n_iter=100, lr=0.01):
        """
        Vectors along the path are optimized by taking average over neirest
        neihbours - 'Mean field approximation'. Vectorized version! 
        This is better since angles have periodicity problem!
        
        """

        vec = np.pad(vecs, ((1, 1),(0, 0)), 'edge')
        m = vec.shape[0] # paded n_rows
        
        for iter in range(n_iter): 
            vec_mean = 0.5*(vec[0:m-2] + vec[2:m])
            dvec = -vec[1:m-1] + vec_mean 
            vec[1:m-1] += lr*dvec
            
            #set flexible boundary
            vec[0] = vec[1] 
            vec[m-1] = vec[m-2]
            
            #vec_mean = (vec[0:m-2] + vec[1:m-1] + vec[2:m] ) / 3.0
            #dvec = -vec[1:m-1] + vec_mean 
            #vec[1:m-1] += lr*dvec

        return  vec[1:m-1] # removed pads
      
    
    
    def Angles2UnitVecs(self, angles):
        """ 
        
        """
        U, V, W = np.sin(angles[:,0])*np.cos(angles[:,1]), \
                  np.sin(angles[:,0])*np.sin(angles[:,1]), \
                  np.cos(angles[:,0])
           
        return np.column_stack([U, V, W])
        
         
    def OptAngles1(self, angles, n_iter=100, lr=0.01):
        """
        Angles are adjusted based on neirest neihbours.
        This is non-optimal fn 
        
        """
        pad = 1
        theta = np.copy(angles[0])
        phi = np.copy(angles[1])
        
        theta = np.pad(theta, (pad, pad), 'edge')
        phi = np.pad(phi, (pad, pad), 'edge')
        
        for iter in range(n_iter): 
            for i in range(1, len(theta) - 1):
                thetamean = 0.5*(theta[i-1] + theta[i+1])
                phimean = 0.5*(phi[i-1] + phi[i+1])
                
                dtheta = -theta[i] + thetamean # modify
                dphi = -phi[i] + phimean
                                
                theta[i] += lr*dtheta
                phi[i] += lr*dphi 
                
        return theta[1:len(theta)-1], phi[1:len(phi)-1]
  
    
    def SelFrames(self, dat, ml_xyz, M, dr, dz):
        """
        Selects frames on a disk surface (both side) based on distances:
            dz - along disk normal
            dr - along radial direction
            
        """

        data = np.copy(dat) - ml_xyz    # translate
        a = np.dot(data, M)             # rotate
        a = a*a; dz = dz*dz; dr = dr*dr # square        
  
        mask = (a[:,2] <= dz) & (a[:,1] + a[:,0] <= dr)
        indxs = np.where(mask)[0]
        
        return indxs

    def SelCells(self, dat, ml_xyzL, ml_xyzM, ml_xyzR, Ml, Mm, Mr, dr):
        """
        Selects frames between two neighbouring milestone planes (aligned to milestone disk)
        and also sorts based on middle disk raduis:
        dz - along disk normal
        dr - along radial direction
            
        """
        dataL = np.copy(dat) - ml_xyzL   # translate left disk and dat with it
        aL = np.dot(dataL, Ml)           # rotate
        
        dataM = np.copy(dat) - ml_xyzL   
        aM = np.dot(dataM, Mm)           
              
        dataR = np.copy(dat) - ml_xyzR   # translate ...
        aR = np.dot(dataR, Mr)           # rotate
               
        maskZ = (aL[:,2] >= 0.0) & (aR[:,2] < 0.0) # choose all points between plances 
        maskR = aM[:,0]**2 + aM[:,1]**2 <= dr**2   # choose points within mid disk dr
        mask = maskZ * maskR                       # combine masks     
        
        indxs = np.where(mask)[0]
        
        return indxs
    
    def SortAllPoints(self, dat, normals, normalsMid, pathP, pathMid, dr, dz, SortMethod='surface'):
        """

        """

        n_disks = normals.shape[0]
        n_cells = normalsMid.shape[0]
        z_hat = (0, 0, 1)             # unit vecor along z
        datS = np.zeros((1,5))        # empty df for appending

        if SortMethod == 'surface':
            for i in range(n_disks):
                M = rotation_matrix(normals[i], z_hat) 
                frame_ids = ConsMile.SelFrames(dat, pathP[i], M, dr = dr, dz = dz)
                diskID = np.ones_like(frame_ids, dtype=int)*i
                datS = np.append(datS, np.column_stack((dat[frame_ids], diskID, frame_ids)), axis=0)
            datS = datS[1:]

        elif SortMethod == 'middle':
            for i in range(n_cells):
                Ml = rotation_matrix(normals[i], z_hat) 
                Mm = rotation_matrix(normalsMid[i], z_hat)
                Mr = rotation_matrix(normals[i+1], z_hat)
                frame_ids = ConsMile.SelCells(dat, pathP[i], pathMid[i], pathP[i+1], Ml, Mm, Mr, dr=dr)
                midIDs = np.ones_like(frame_ids, dtype=int)*i
                datS = np.append(datS, np.column_stack((dat[frame_ids], midIDs, frame_ids)), axis=0)
            datS = datS[1:]

        else:
            print('Specify one of the available methods to sort data points!')

        return datS

if __name__=='__main__':
       
    #from SmoothPath import BuildReactPath, Generate3Data
    #from InterpolateCurve import InterpolatePath

    dim = 3
    n_points = 10000
    seed = 10
    n_bins = 20
    pd = 0.6
    ml_length = .6
    rad = 1.2
    n_iter = 500
    plot_cellP = True
    
    dat = Generate3Data(n_points)
    ReactPath = BuildReactPath(dat, n_bins=n_bins, n_samples=2, w_size=5, dim=dim)   
    points = ReactPath.GetReactPath(rseed=seed)
    
    pathAll = InterpolatePath(points, pair_dist=ml_length/2, dim=dim, kind='linear')
    mp = pathAll.shape[0]
    midID = [i for i in range(1,mp-1,2)]
    pID = [i for i in range(0,mp,2)] 
    pathMid = pathAll[midID]    
    pathP = pathAll[pID]    
   
    # construct 3D milestones (disks)
    ConsMile = ConstructMilestones3D(pathAll)#, ml_length = ml_length/2)
    angles = ConsMile.GetAngles()
    #angles = np.array(angles)
    angles = ConsMile.OptAngles(angles, n_iter=n_iter)
    #print(angles.shape)
      
    U, V, W = ml_length*np.sin(angles[:,0])*np.cos(angles[:,1]), \
              ml_length*np.sin(angles[:,0])*np.sin(angles[:,1]), \
              ml_length*np.cos(angles[:,0])
       

    normalsAll = np.column_stack([U, V, W])
    normals = normalsAll[pID]    
    normalsMid = normalsAll[midID]
    n_circle = normals.shape[0]
    n_circleMid = normalsMid.shape[0]

   
    fig = plt.figure(figsize = [16,16])
    ax = plt.axes(projection='3d')
    #p = ax.scatter3D(dat[:,0], dat[:,1], dat[:,2], c = range(int(len(dat))), alpha=0.05)
    #ax.plot3D(pathP[:,0], pathP[:,1], pathP[:,2], 'red', marker='o', markersize=0.5, alpha=0.3)
    
    SelIndx = []
    for i in range(n_circle):
        c = Circle((0,0), rad, facecolor='grey', alpha=0.4)
        ax.add_patch(c)
        M = rotation_matrix(normals[i], (0, 0, 1)) 
        pathpatch_2d_to_3d(c, pathP[i], normal = normals[i])
        S1 = ConsMile.SelFrames(dat, pathP[i], M, dr=1., dz=0.1)
        dat1 = dat[S1]
        if plot_cellP != True:
            p = ax.scatter3D(dat1[:,0], dat1[:,1], dat1[:,2], alpha=0.5, s=1.8)
     
    #plot selected pints on each cell
    if plot_cellP == True:
        for i in range(1,n_circle+n_circleMid,2):
            
            Ml = rotation_matrix(normalsAll[i-1], (0, 0, 1)) 
            Mm = rotation_matrix(normalsAll[i], (0, 0, 1))
            Mr = rotation_matrix(normalsAll[i+1], (0, 0, 1))
            
            S2 = ConsMile.SelCells(dat, pathAll[i-1], pathAll[i], pathAll[i+1], Ml, Mm, Mr, dr=10*rad)
            print('milestone' + str(i), S2)
            datM1 = dat[S2] #select
            p = ax.scatter3D(datM1[:,0], datM1[:,1], datM1[:,2], alpha=0.5, s=1.8)
            #ml = np.ones_like(S2)*i
            #datM1 = np.column_stack((datM1, ml)) #append mlid as column
        
        
    #ax.quiver(pathP[:,0], pathP[:,1], pathP[:,2], U, V, W, color='magenta')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3') 
    
    x_min, y_min, z_min = pathP.min(axis=0) 
    x_max, y_max, z_max = pathP.max(axis=0)
    
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min - 3, y_max + 3)
    ax.set_zlim3d(z_min - 3, z_max + 3)
            
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    
    #set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w'); ax.yaxis.pane.set_edgecolor('w'); ax.zaxis.pane.set_edgecolor('w')
    #ax.grid(False)
    #fig.colorbar(p, ax=ax)
    plt.show() 
