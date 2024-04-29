import numpy as np

def StateLabels_1D(rmsd, r_min=2., r_max=15.0, bin_size=0.4):
    """
    Function assignes state labels to all frames based on
    its location in rmsd (1D) space. 
    ------------------------
    Input:
        rmsd     - rmsd values in a 1D array shape
        r_min    - lower end for binning
        r_max    - higher end for binning 
        bin_size - bin size for intervals, distance between 1D milestones
        
    Output:
        out      - state labels for all data points (n,3) 
                  1st column - rmsd values
                  2nd column - state labels (integers)
                  3rd column - array indexes (this is used to put everything back
                  to original order)
    """

    n_bins = int((r_max - r_min) / bin_size)
    out = np.zeros((1,3))        # empty df for appending
    
    for i in range(n_bins+1):        
        x_left = r_min + i*bin_size 
        x_right = r_min + (i+1)*bin_size 
        mask = (rmsd >= x_left) & (rmsd < x_right)
    
        frame_ids = np.where(mask)[0]
        labelI = np.ones_like(frame_ids, dtype=int)*i
        out = np.append(out, np.column_stack((rmsd[frame_ids], labelI, frame_ids)), axis=0)
    
    out = out[1:] #rm first row
    out = out[out[:,2].argsort()] #sort based on frame_indx

    return out
