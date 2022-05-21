import sys
import os
import numpy as np

class ConstructMilestones:
   
    def __init__(self, path, ml_length):
        """
        
        """
        self.path = path
        self.ml_length = ml_length

        
    def GetSlopes(self):
        """
        Simply get slopes of normals alog the path from 
        norm_slope = -line_slope
        """
        fp = self.path
        #n_rows, n_cols = fp.shape[0], fp.shape[1]
        #slope = np.zeros(shape = (n_rows, n_cols-1))
        #right = fp[1:n_rows]
        #left = fp[0:n_rows-1]
        for i in range(len(fp)-1):
            slope[i] = -(fp[i+1,0] - fp[i,0])/(fp[i+1,1] - fp[i,1])
        slope[-1]=slope[-2]
        
        return slope
    
    
    def Slope2XY(self, slopes):
        """
        Calculate head and tail positions of milestones(lines normal to path)
        from path positions
        """
        
        X = self.path[:,0]; Y = self.path[:,1]
        
        dX = 0.5 * self.ml_length / np.sqrt(1.0 + slopes**2)
        Xp = X + dX          ;  Xn = X - dX
        Yp = Y + dX * slopes ;  Yn = Y - dX * slopes
    
        return (Xp, Xn, Yp, Yn)

        
    def Angle2XY(self, angles):
        """
        Calculate head and tail positions of milestones(lines normal to path)
        given angles in 2D
        """
        X = self.path[:,0]; Y = self.path[:,1]
        
        dX = 0.5 * self.ml_length * np.cos(angles*np.pi/180)
        dY = 0.5 * self.ml_length * np.sin(angles*np.pi/180)
        Xp = X + dX; Xn = X - dX
        Yp = Y + dY; Yn = Y - dY
    
        return (Xp, Xn, Yp, Yn)

    
    def CorrectXYnp(self, Xp, Yp, Xn, Yn):
        """
        Makes sure all heads (of norm vectors) are on the same side of path,
        which is needed for efficient optimization later.
        """
        slopes = (Yp - Yn) / (Xp - Xn)
        ang = np.arctan(slopes)*180/np.pi
        
        if ang[0] > 0.0:
            addA = 180.0
        else:
            addA = -180.0
        
        for i in range(1,len(Xp)):
            d1 = (Xp[i] - Xp[i-1])**2 + (Yp[i] - Yp[i-1])**2
            d2 = (Xp[i] - Xn[i-1])**2 + (Yp[i] - Yn[i-1])**2
            if d1 > d2:
                Xn[i], Xp[i] = Xp[i], Xn[i]
                Yn[i], Yp[i] = Yp[i], Yn[i]
                ang[i] = ang[i] + addA

        return (Xp, Yp, Xn, Yn, ang)
    
    

    
    def OptAngles(self, angles, n_iter=10, lr=0.01):
        """
        Angles are adjusted based on neirest neihbours.
        This modules requires all vector pointing to the same side of path,
        which makes it efficient.
        
        """
        pad = 1
        theta = np.copy(angles)
        theta = np.pad(theta, (pad, pad), 'edge')
        
        for iter in range(n_iter): 
            for i in range(1, len(theta) - 1):
                amean = 0.5*(theta[i-1] + theta[i+1])
                dtheta = -theta[i] + amean
                theta[i] += lr*dtheta
                
        return theta[1:len(theta)-1]
    
   
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
    


if __name__=='__main__':
       
    import matplotlib.pyplot as plt
    from SmoothPath import BuildReactPath, Generate2Data, Generate3Data

    dim=2
    n_points = 2000
    seed = 10
    n_bins = 25


    if dim==2:
        dat = Generate2Data(n_points)
    else:
        dat = Generate3Data(n_points)
        
    ReactPath = BuildReactPath(dat, n_bins=n_bins, n_samples=2, w_size=5, dim=dim)   
    points = ReactPath.GetReactPath(rseed=seed)
    
    ml_length = 20
    ConsMile = ConstructMilestones(dat, ml_length = ml_length)
    slope = ConsMile.GetSlopes()
    
    
    fig = plt.figure(figsize = [8,6])
    if dim==2:
        plt.plot(dat[:,0], dat[:,1], ls = '', marker = 'o', markersize = 4, alpha=0.1)
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
