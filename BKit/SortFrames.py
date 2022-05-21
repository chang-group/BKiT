import sys
import os
import numpy as np
from ConstructMilestones3D import rotation_matrix
 
class SortFrames:
   
    def __init__(self, dat, dr, dz):
        """
        
        """
        self.dat = dat
        self.dr = dr
        self.dz = dz
                
          
    def SelFrames(self, ml_xyz, M):
        """
        Selects frames on a disk surface (both side) based on distances:
            dz - along disk normal
            dr - along radial direction
            
        """
        data = np.copy(self.dat) - ml_xyz                     # translate
        a = np.dot(data, M)                                   # rotate
        a = a*a; dz = self.dz*self.dz; dr = self.dr*self.dr   # square        
        mask = (a[:,2] <= dz) & (a[:,1] + a[:,0] <= dr)
        indxs = np.where(mask)[0]

        return indxs

    def SelCells(self, ml_xyzL, ml_xyzM, ml_xyzR, Ml, Mm, Mr):
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
        maskR = aM[:,0]**2 + aM[:,1]**2 <= self.dr**2   # choose points within mid disk dr
        mask = maskZ * maskR                            # combine masks     
        
        indxs = np.where(mask)[0]
        
        return indxs
    
    def SortAllPoints(self, normals, normalsMid, pathP, pathMid, SortMethod='surface'):
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
                frame_ids = self.SelFrames(pathP[i], M)
                diskID = np.ones_like(frame_ids, dtype=int)*i
                datS = np.append(datS, np.column_stack((self.dat[frame_ids], diskID, frame_ids)), axis=0)
            datS = datS[1:]

        elif SortMethod == 'middle':
            for i in range(n_cells):
                Ml = rotation_matrix(normals[i], z_hat) 
                Mm = rotation_matrix(normalsMid[i], z_hat)
                Mr = rotation_matrix(normals[i+1], z_hat)
                frame_ids = self.SelCells(pathP[i], pathMid[i], pathP[i+1], Ml, Mm, Mr)
                midIDs = np.ones_like(frame_ids, dtype=int)*i
                datS = np.append(datS, np.column_stack((self.dat[frame_ids], midIDs, frame_ids)), axis=0)
                CellIndx[frame_ids] = i

            datS = datS[1:]

        else:
            print('Specify one of the available methods to sort data points!')

        return datS, CellIndx

if __name__=='__main__':
    print('smile :)')
         
    
