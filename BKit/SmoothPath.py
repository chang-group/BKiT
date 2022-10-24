import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KDTree


class BuildSmoothMeanPath:
    """
        This module builds a reaction path(line) in a PCA
        space. 

        Parameters
        ----------
        dat : np.ndarray (n_rows, n_cols)
            Input data (Principal components)
        w_size : float
            window size for rolling average
        stride : 
            stride
            
        Returns
        -------
        Selected data points that represent choosen path

        """        

    def __init__(self, dat, w_size, stride, thresh=-0.2, rm_iters=20):
        
        self.w_size = w_size
        self.stride = stride
        self.thresh = thresh
        self.dat = dat
        self.rm_iters = rm_iters
                
    def GetPath(self):
         """
         Construct path based on simple rolling average
         """
         path = self.roll_ave(self.w_size, self.stride)
         points = self.rm_sharp(path)
         return points

    def GetPathKD_ReOpt(self, dat, mean_path, rad, w_size, stride):
        """
        Construct path based on local neighbourhood distance
        """
                
        tree = KDTree(dat, leaf_size = 10) 
        n_p = mean_path.shape[0]

        points = []
        for i in range(n_p):
            selected_indx = tree.query_radius(mean_path[i:i+1], r=rad)[0]
            if selected_indx.size != 0: 
                sample = dat[selected_indx].mean(axis=0)  
                points.append(sample)
                       
        points = np.array(points)
        points = self.rm_sharp(points)

        return points

    def GetPathKD(self, rad, w_size, stride):
        """
        Construct path based on local neighbourhood distance
        """
        
        tree = KDTree(self.dat, leaf_size = 10) 
        mean_path = self.roll_ave( w_size, stride=stride)
        n_p = mean_path.shape[0]

        points = []
        for i in range(n_p):
            selected_indx = tree.query_radius(mean_path[i:i+1], r=rad)[0]
            if selected_indx.size != 0: 
                sample = self.dat[selected_indx].mean(axis=0)  
                points.append(sample)
                       
        points = np.array(points)
        points = self.rm_sharp(points)

        return points


    def roll_ave(self, w_size, stride):
        """
        Rolling average
        """
        
        pad = w_size//2
        data_new = np.copy(self.dat)
        data_new = np.pad( data_new, pad_width= ((pad, pad), (0, 0)), mode = 'edge' )
        N = data_new.shape[0]
        r_mean = [] 
        
        for i in range(0, N - w_size + 1, stride):
            ave = data_new[i: i + w_size].mean(axis=0)
            r_mean.append(ave)
    
        return np.array(r_mean)
    
    
    def rm_sharp(self, points):
        """
        Remove points based on how sharp the corner is
        """
        
        for it in range(self.rm_iters):
            mask = np.ones(points.shape[0], dtype=bool)
            for i in range(1, len(points) - 1):
            
                a =  points[i+1] - points[i]
                b =  points[i-1] - points[i]
                a /= np.linalg.norm(a)
                b /= np.linalg.norm(b)
                cosA = np.dot(a,b)
                mask[i] = cosA < self.thresh
            
            new_points = points[mask].copy() 
            points = new_points.copy()
                        
        return points
    
    def sm_sharp(self, points):
        """
        Smooth corners based on how sharp the corner is by moving every
        point to center of triangle
        
        """
        
        pnts = points.copy()
        for i in range(1, len(pnts) - 1):
            
            a =  pnts[i+1] - pnts[i]
            b =  pnts[i-1] - pnts[i]
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            cosA = np.dot(a,b)
            if cosA < self.thresh:
                pnts[i] = (pnts[i-1] + pnts[i] + pnts[i+1]) / 3.0
                #pnts[i] = (pnts[i-1] + pnts[i+1]) / 2.0
        
        return pnts
   
     

if __name__=='__main__':
    
    plot2D = False
    n_points = 500
    seed = 10
    w_size = 100
    stride = 20

    from Utils import Generate2Data, Generate3Data

    if plot2D:
        dat = Generate2Data(n_points)
    else:
        dat = Generate3Data(n_points)

    ReactPath = BuildSmoothMeanPath(dat, w_size=w_size, stride=stride)    
    points = ReactPath.GetPath()
    #points = ReactPath.GetPathKD(rad=1.0, w_size=w_size, stride=stride)
   
    
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
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3') 
        fig.colorbar(p, ax=ax)
        plt.show()
