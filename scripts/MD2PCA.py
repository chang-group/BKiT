import sys
import pytraj as pt
from sklearn.decomposition import PCA
import numpy as np

##################################################################################
#specify path of promtop and dcd files
path = '../CDK8/PL4_OH/data/'
TOP = path + 'GB.prmtop'
pdb_path = path + '0.pdb'
refpdb = pt.load(pdb_path) 
mask_align = "(:1-359)&(@N,@C,@CA,@O)" 
mask_selec = ['@CA','!@H*']
DIR = 'output/'
out_dir = DIR + 'PCA/'
inp_dir = path + 'kinetics/'

eigvecs = np.load(DIR + '/EigVecs.npy')
refframe = np.load(DIR + '/RefFrame.npy')

n_res = 620
maskS1 = mask_selec[0] + ",:" + str(n_res)
maskS2 = mask_selec[1] 

#############################################################################################
first_mdID = int(sys.argv[1])   # First index of the short MD simulation folder (e.g. i=0 for P0)
last_mdID  = int(sys.argv[2])   # Last index of the short MD simulation folder (e.g. i=6000 for P6000)
freq       = 10                 # frequencing of the folder index, e.g. frq=10 if your folders are P0, P10, .
first_repID= 1                  # first replica in the above P* folder
last_repID = 20                 # last replica  in the above P* folder

for i in range(first_mdID, last_mdID+freq, freq):
    print("Interval - ", i)
    pca = []
    for j in range(first_repID, last_repID + 1):
        path_md = inp_dir + 'P' + str(i) + '/02.MD' + str(j) + '/MD1_unwrap.dcd'
        print(path_md)
        mdtraj = pt.load(path_md, TOP)
        pt.superpose(mdtraj, ref=refpdb, mask=mask_align)
        md = mdtraj[maskS1][maskS2]
        md = md.xyz.reshape(md.n_frames, md.n_atoms*3) - refframe

        pca.append(np.column_stack([np.dot(md, eigvecs[:,0]),
                                    np.dot(md, eigvecs[:,1]),
                                    np.dot(md, eigvecs[:,2])]))
        md = None # clear garbage collector

    ########### I/O projections to a file #########
    print("writing outputs to " + out_dir)
    PCA = np.concatenate(pca, axis=0)
    np.save(out_dir + 'PCA' + str(i) + '.npy', PCA)
        
print("Done")
