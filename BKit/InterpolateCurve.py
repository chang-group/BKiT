import numpy as np
from scipy.interpolate import interp1d

def InterpolatePath(data, pair_dist, dim=2, kind='cubic'):
    """
    Path interpolation applied on cumulative distance that 
    works with n-dimensional input.
    
    """
    
    n_points = data.shape[0]
    dr = data[1:n_points] - data[0:n_points-1]
    cum_dist = np.sqrt((dr*dr).sum(axis=1)).cumsum()
    cum_dist = np.append(0.0,cum_dist)
    pts = int(cum_dist[-1] / pair_dist) 

    func = []
    for idim in range(dim):
        func.append(interp1d(cum_dist, data[:, idim], kind = kind))

    xnew = np.linspace(0, cum_dist[-1], num=pts, endpoint=True)
    out = np.zeros(shape=(len(xnew), dim))
    for idim in range(dim):
        out[:,idim] = func[idim](xnew)        
    
    return out


def InterpolatePathOld(data, pair_dist, dim=2, kind='cubic'):
    """
    Path interpolation applied on cumulative distance that 
    works with n-dimensional input.
    
    """
    
    cum_euc_dist = [0.0] 
    dist = 0
    func = []

    for i in range(len(data)-1):
        dr = data[i] - data[i+1]
        dr2 = (dr*dr).sum()
        dist += np.sqrt(dr2) 
        cum_euc_dist.append(dist)
 
    cum_euc_dist = np.array(cum_euc_dist) 
    pts = int(cum_euc_dist[-1] / pair_dist) 
    
    for idim in range(dim):
        func.append(interp1d(cum_euc_dist, data[:, idim], kind = kind))

    xnew = np.linspace(0, cum_euc_dist[-1], num=pts, endpoint=True)

    out = np.zeros(shape=(len(xnew), dim))
    for idim in range(dim):
        out[:,idim] = func[idim](xnew)        
   
        
    if out.shape[0]%2 == 0:#if even, drop last ml
        out = out[0:-1] 
    
    return out


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    n = 100
    dim = 3
    chunk_length = 2
    x = np.linspace(0, 20, n)
    #x[10]=-3.0
    
    y = np.sin(3*x) - x + (np.random.rand(n) -0.5)  
    z = x - np.cos(y) + (np.random.rand(n) -0.5)
    
    if dim == 2:
        data = np.column_stack([x,y])
    else:
         data = np.column_stack([x,y,z])
         
    dataI = InterpolatePath(data, pair_dist=chunk_length, dim=dim, kind='cubic')

    fig = plt.figure(figsize = [8,6])
    if dim == 2:
        plt.plot(x, y, ls='', marker='o', markersize=3)
        plt.plot(dataI[:,0], dataI[:,1])
        plt.xlabel('X'); plt.ylabel('Y')
        plt.show()
    else:
        ax = plt.axes(projection='3d')
        p = ax.scatter3D(data[:,0], data[:,1], data[:,2] ,alpha=0.3)
        ax.plot3D(dataI[:,0], dataI[:,1], dataI[:,2], 'red')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z') 
        fig.colorbar(p, ax=ax)
        plt.show()
    
    
    
    
