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
  
    


class SortFrames:
   
    def __init__(self, dat, dz):
        """
        
        """
        self.dat = dat
        self.dz = dz
                
          
    def SelFrames(self, ml_xyz, M, dr):
        """
        Selects frames on a disk surface (both side) based on distances:
            dz - along disk normal
            dr - along radial direction
            
        """
        data = np.copy(self.dat) - ml_xyz                     # translate
        a = np.dot(data, M)                                   # rotate
        a = a*a; dz = self.dz*self.dz; dr = dr*dr   # square        
        mask = (a[:,2] <= dz) & (a[:,1] + a[:,0] <= dr)
        indxs = np.where(mask)[0]

        return indxs

    def SelCells(self, ml_xyzL, ml_xyzM, ml_xyzR, dr, Ml, Mm, Mr):
        """
        Selects frames between two neighbouring milestone planes
        (aligned to milestone disk) and also sorts based on middle disk raduis:
        dz - along disk normal
        dr - along radial direction
            
        """
        dataL = np.copy(self.dat) - ml_xyzL   # translate left disk and dat with it
        aL = np.dot(dataL, Ml)                # rotate
        
        dataM = np.copy(self.dat) - ml_xyzL   
        aM = np.dot(dataM, Mm)           
              
        dataR = np.copy(self.dat) - ml_xyzR   # translate ...
        aR = np.dot(dataR, Mr)                # rotate
               
        maskZ = (aL[:,2] >= 0.0) & (aR[:,2] < 0.0)      # choose all points between plances 
        maskR = aM[:,0]**2 + aM[:,1]**2 <= dr**2   # choose points within mid disk dr
        mask = maskZ * maskR                            # combine masks     
        
        indxs = np.where(mask)[0]
        
        return indxs
    
    def SortAllPoints(self, normals, normalsMid, dr, pathP, pathMid, SortMethod='surface'):
        """

        """

        n_disks = normals.shape[0]
        n_cells = normalsMid.shape[0]
        z_hat = (0, 0, 1)             # unit vecor along z
        datS = np.zeros((1,5))        # empty df for appending

        nf = self.dat.shape[0] 
        CellIndx = np.ones(nf, dtype=int)*1000 # Index assigned to regions outside of the cells


        if SortMethod == 'surface':
            for i in range(n_disks):
                M = rotation_matrix(normals[i], z_hat) 
                frame_ids = self.SelFrames(pathP[i], M, dr[i])
                diskID = np.ones_like(frame_ids, dtype=int)*i
                datS = np.append(datS, np.column_stack((self.dat[frame_ids], diskID, frame_ids)), axis=0)
            datS = datS[1:]

        elif SortMethod == 'middle':
            for i in range(n_cells):
                Ml = rotation_matrix(normals[i], z_hat) 
                Mm = rotation_matrix(normalsMid[i], z_hat)
                Mr = rotation_matrix(normals[i+1], z_hat)
                frame_ids = self.SelCells(pathP[i], pathMid[i], pathP[i+1], dr[i], Ml, Mm, Mr)
                midIDs = np.ones_like(frame_ids, dtype=int)*i
                datS = np.append(datS, np.column_stack((self.dat[frame_ids], midIDs, frame_ids)), axis=0)
                CellIndx[frame_ids] = i

            datS = datS[1:]

        else:
            print('Specify one of the available methods to sort data points!')

        return datS, CellIndx




    
if __name__=='__main__':
       
    from Utils import Generate3Data, Generate2Data, pathpatch_2d_to_3d
    from BKit import BuildSmoothMeanPath 
    from BKit import InterpolatePath

    plot2D = True
    w_size = 100
    stride = 20
    n_points = 500
    dr = .8
    alpha = .5
    fs = 12 
        
    if plot2D:
        dat = Generate2Data(n_points)
    else:
        dat = Generate3Data(n_points)
        
    ReactPath = BuildSmoothMeanPath(dat, w_size=w_size, stride=stride)    
    points = ReactPath.GetPath()
    
    ConsMile = ConstructMilestones3D(points)
    vecs = ConsMile.GetVectors()
    normals = ConsMile.OptVectors(vecs, n_iter=1000, lr=0.01)

    n_disks = normals.shape[0]
    rad = np.ones(n_disks)*dr
    
    fig = plt.figure(figsize = [8,6])
    
    if plot2D:
        plt.plot(dat[:,0], dat[:,1], ls = '', marker = 'o', markersize = 4)
        plt.plot(points[:,0], points[:,1], ls = '-', marker = 'o', markersize = 4)
    
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.show()
      
    else:
    
        ax = plt.axes(projection='3d')
        p = ax.scatter3D(dat[:,0], dat[:,1], dat[:,2], c = range(int(len(dat))), alpha=0.2)
        ax.plot3D(points[:,0], points[:,1], points[:,2], 'red', marker='o')
        
        for i in range(n_disks):
            c = Circle((0,0), rad[i], facecolor='grey', alpha=alpha)
            ax.add_patch(c)
            pathpatch_2d_to_3d(c, points[i], normal = normals[i])
        
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3') 
        fig.colorbar(p, ax=ax)
        plt.show()
    
    
    