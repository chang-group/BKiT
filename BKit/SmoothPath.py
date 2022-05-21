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

    def __init__(self, dat, w_size, stride, thresh=-0.2, rm_iters=20, dim=2):
        
        self.w_size = w_size
        self.stride = stride
        self.dim = dim
        self.thresh = thresh
        self.dat = dat
        self.rm_iters = rm_iters
                
    def GetPath(self):
         """

         """
         path = self.roll_ave(self.w_size, self.stride)
         points = self.rm_sharp(path)
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
    

#####################################################################################################


class BuildReactPath:
    """
        This module builds a reaction path(line) in a PCA
        space given sampled data values. Ligand dissociation 
        data is usually imbalanced. Thus, balanced sampling is used
        in combination with KDTree(efficient neighbour search) to
        construct the path. Method supports any dimention, and different path
        can be constructed by changing random seed.

        Parameters
        ----------
        dat : np.ndarray (n_rows, n_cols)
            Input data (Principal components)
        n_bins : int
            number of bins for binning of PCA1
        n_samples : int
            Number of sample points to pick from each bin. Note: If n_samples 
            is more than min samples, then it is set to min samples(2).

        Returns
        -------
        Selected data points that represent choosen path

        """        

    def __init__(self, dat, n_bins, n_samples, w_size, dim=2):
        
        self.tree = KDTree(dat, leaf_size = 10) 
        self.dat = dat
        self.n_samples = n_samples
        self.w_size = w_size
        self.dim = dim
        
        bin_size = (dat[:,0].max() - dat[:,0].min()) / n_bins
        histPC1 = np.histogram(dat[:,0], bins = n_bins)
        self.freqs = histPC1[0]
        self.neighb_rad = 1.0 * bin_size

        
    def SampleData(self, n_samples=2, rseed=10):
        """
        Here we apply balanced sampling on imbalanced data. Bin frequencies
        are used a max sample size to pick samples from. It assumes initial
        n points are located in 1st bin and etc..

        Parameters
        ----------
        n_samples : int, optional
            Number of data points to pick from each bin. The default is 10
        rseed : int, optional
            Random seed for generating different reaction paths. The default is 10.

        Returns
        -------
        frame_ids: int, numpy.nd
            Sampled frame ids from bins
        
        """
        np.random.seed(rseed)
        n_min_samples = self.freqs.min()
        if n_samples > n_min_samples:
            n_samples = n_min_samples
        
        accum = -1
        frame_ids = []
        for i, f in enumerate(self.freqs):

            if f < n_min_samples: 
                raise ValueError('Each bin must have at least one sample! Reduce n_bins!') 
                
            #if i == 0: #make sure bound state has more samples
            #    num_samples = 3*n_samples
            #else:
            #    num_samples = n_samples
                
            ids = np.random.randint(accum + 1, accum + f + 1,  n_samples)
            accum += f
            frame_ids.append(ids)

        return np.array(frame_ids)

        
    def GetReactPath(self, rseed):
        """
        Parameters
        ----------
        path_list : TYPE
            DESCRIPTION.
        rot : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
                
        inds = self.SampleData(n_samples=self.n_samples, rseed=rseed)
        inds = inds.flatten()
        mean_dat = self.roll_ave( self.w_size, stride=1)
        inds.sort()

        points = []
        x0 = 0.0 
        for i in inds:
            
            # Sample from bins with KDTree
            #sample = self.dat[self.tree.query_radius(self.dat[i:i+1], r = self.neighb_rad)[0]].mean(axis=0) 
            
            # Sample from bins only
            #sample = mean_dat[i]  
            
            # Sample from bins along mean path with KDTree (best!). Mean path used as reference
            sample = mean_dat[self.tree.query_radius(mean_dat[i:i+1], r = self.neighb_rad)[0]].mean(axis=0)  

            #encourage only positive dx
            dx = sample[0] - x0 
            if dx >= 0.0:
                points.append(sample)
                x0 = sample[0]
        
        points = np.array(points)
    
        #add a point from bound state
        npo = 1
        boundID = np.random.randint(5)
        BoundPoint = self.dat[self.tree.query_radius(self.dat[boundID: boundID+npo], r=self.neighb_rad)[0]].mean(axis=0).reshape(npo, self.dim)
        points = np.concatenate([BoundPoint,points])
       
        #points = np.concatenate([self.dat[boundID:boundID + 2],points])
        
        
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
    
    
    def rm_sharp(self, points, rm_iters, thresh=-0.1):
        """
        Remove sharp turns? Not needed for now!
        """
        
        for it in range(rm_iters):
            mask = np.ones(points.shape[0], dtype=bool)
            for i in range(1, len(points) - 1):
            
                a =  points[i+1] - points[i]
                b =  points[i-1] - points[i]
                a /= np.linalg.norm(a)
                b /= np.linalg.norm(b)
                cosA = np.dot(a,b)
                mask[i] = cosA < thresh
            
            new_points = points[mask].copy() 
            points = new_points.copy()
      
        return points
    
    def sm_sharp(self, points):
        """
        Remove sharp turns? Not needed for now!
        """
        pnts = points.copy()
        for i in range(1, len(pnts) - 1):
            
            a =  pnts[i+1] - pnts[i]
            b =  pnts[i-1] - pnts[i]
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            cosA = np.dot(a,b)
            if cosA < 0.0:
                pnts[i] = (pnts[i-1] + pnts[i] + pnts[i+1]) / 3.0

        
        return pnts
    
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

if __name__=='__main__':
    
    plot2D = False
    n_points = 1000
    seed = 10

    if plot2D:
        dat = Generate2Data(n_points)
    else:
        dat = Generate3Data(n_points)

    #ReactPath = BuildReactPath(dat, n_bins=25, n_samples=3, w_size=5)   
    #points = ReactPath.GetReactPath(rseed=seed)
    
    ReactPath = BuildSmoothMeanPath(dat, w_size=100, stride=40)    
    points = ReactPath.GetPath()
   
    
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
