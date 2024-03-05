import numpy as np


class TransitionKernel:
   
    def __init__(self, traj_size, OutCellID, check_escape):
        """
        
        """
        self.T = traj_size
        self.outID = OutCellID
        self.check_escape = check_escape

        
    def KernelPerTraj(self, trajID, trajI):
        """
                
        General method to count transitions: All transitions between
        neighbouring milestones are considered.
        
          
        Parameters
        -------
        trajID : TYPE -- int
            Short MD trajectory index

        trajI : np.array(shape=(traj size), int) 
            state labels per trajectory

        Returns
        -------
        trans : np.array(shape=(number of transitions, 5), int) 
            Short MD trajectory index
        
        count_crossing : TYPE -- int
            total number of transions per short MD


        """
        
        dt = 1
        #frameid = np.arange(0, self.T-dt)             # frame indx per trajectory
        frameid = np.arange(0, len(trajI)-dt)          # frame indx per trajectory
        transition = trajI[dt:] - trajI[:-dt]          # 1 step diff --> tells if crossing 
                                                       # happend within 1 time step (1000 md steps)
        frameid_trans = frameid[abs(transition) == 1] 
        milestone = np.max(np.array([trajI[frameid_trans], trajI[frameid_trans + 1]]).T, axis=1) 
        
        trans = np.zeros((1,5), dtype=int)
        count_crossing = 0
        
        if len(milestone) > 0 :
            m_ini, t_ini = milestone[0], frameid_trans[0]

            for i in range(len(milestone)-1):
    
                if (abs(milestone[i+1] - milestone[i]) == 1):
                #if (abs(milestone[i+1] - milestone[i]) > 0):

                    m_end, t_end = milestone[i+1], frameid_trans[i+1]
            
                    if m_ini != milestone[i]:
                        m_ini,t_ini = milestone[i],frameid_trans[i]

                    if self.check_escape:
                        escape = (np.any(trajI[t_ini:t_end] == self.outID))
                    else:
                        escape = False

                    if escape == False:                  
                        tmp = np.array([trajID, m_ini, m_end, t_ini, t_end], dtype=int).reshape(1,5)
                        trans = np.append(trans, tmp, axis=0)
                        count_crossing += 1
                        m_ini, t_ini = m_end, t_end
                                         
                else:
                    continue

        return trans[1:], count_crossing
    

    def KernelPerTrajS(self,trajID, trajI):
        """
        Strict method to count transitions: Only local neareast neighbour
        transitions are allowed. Trajectory with conformations starting
        outside the path are ignored. Part of the trajectory after jump is also
        ignored. Conformations that return to same spot in milestone is believed to
        be a different conformation due to accounting for a few PC components. 
          
        Parameters
        -------
        trajID : TYPE -- int
            Short MD trajectory index

        trajI : np.array(shape=(traj size), int) 
            state labels per trajectory

        Returns
        -------
        trans : np.array(shape=(number of transitions, 5), int) 
            Short MD trajectory index
        
        count_crossing : TYPE -- int
            total number of transions per short MD

        """

        
        check_escape = True
        dt = 1
        frameid = np.arange(0, len(trajI)-dt)          # frame indx per trajectory
        transition = trajI[dt:] - trajI[:-dt]          # 1 step diff --> tells if crossing 
                                                       # happend within 1 time step (1000 md steps)
        frameid_trans = frameid[abs(transition) == 1] 
        milestone = np.max(np.array([trajI[frameid_trans], trajI[frameid_trans + 1]]).T, axis=1) 
        
        trans = np.zeros((1,5), dtype=int)
        count_crossing = 0
        
        if len(milestone) > 0 :
            m_ini, t_ini = milestone[0], frameid_trans[0]

            for i in range(len(milestone)-1):
    
                if (abs(milestone[i+1] - milestone[i]) == 1):
                    m_end, t_end = milestone[i+1], frameid_trans[i+1]

                    if m_ini != milestone[i]:
                        m_ini,t_ini = milestone[i],frameid_trans[i]
                    
                    if check_escape:
                        escape = (np.any(trajI[t_ini:t_end] ==  self.outID))
                    else:
                        escape = False

                    if escape == False:                  
                        tmp = np.array([trajID, m_ini, m_end, t_ini, t_end], dtype=int).reshape(1,5)
                        trans = np.append(trans, tmp, axis=0)
                        count_crossing += 1
                        m_ini, t_ini = m_end, t_end
                                         
                else:
                    continue

        return trans[1:], count_crossing

    
    def AllTrans(self, cell_ids, frac=1.0, count_method='orig'):
        """
                
        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
        transI : TYPE
            DESCRIPTION.

        """
        
        ntraj = int(len(cell_ids) / self.T) 
        print('number of trajs', ntraj)
        TR = np.zeros((1,5), dtype=int)
        Ncross=0
        
        for i in range(0, ntraj, 1):
            left = i * self.T
            right = i * self.T + int(self.T *frac)
            trajI = cell_ids[left:right]             

            if count_method == 'orig':
                trans_trajI, Icross = self.KernelPerTraj(i, trajI)
                TR = np.append(TR, trans_trajI, axis=0)
                Ncross += Icross
            elif (count_method == 'strict') & (trajI[0] !=  self.outID):
                trans_trajI, Icross = self.KernelPerTrajS(i, trajI)
                TR = np.append(TR, trans_trajI, axis=0)
                Ncross += Icross
            else:
                pass
                
        #if len(milestone) < 1:
        #    raise ValueError('Trajectory ' + str(trajID) + ' missing transitions! ' +
        #                     'It results in incorrect Free E calculation! ' + 
        #                     'Consider changing transition path.')
        
        print('Total transitions = ', Ncross)
        return TR[1:], Ncross
    
    
    def Kmat_time(self, trans, nM, norm=True):
        """
        transition matrix and mean first passage time....
    
        """
        kmat = np.zeros((nM,nM), dtype=int)
        time = [[] for i in range(nM)]
        mfpt = np.zeros((1,3), dtype=int)

        for i in range(len(trans)):
            m_ini, m_f = trans[i,1], trans[i,2]
            t = trans[i,4] - trans[i,3] 
    
            kmat[m_ini-1, m_f-1] += 1
            time[m_ini-1].append(t)


        mfpt = []
        for i in range(len(time)):
            tmp = np.array(time[i])

            if len(tmp) != 0: # if list is not empty
                mfpt.append([i, tmp.mean(), tmp.std()])

        if norm:
            kmat = (kmat.T / kmat.sum(axis=1)).T

        return kmat, mfpt

            
if __name__=='__main__':

    DIR_SAVE = '../notebooks/output/'
    print("reading inputs from " + DIR_SAVE)
    MIDX = np.load(DIR_SAVE + 'CellIndx.npy')
    T = 1000
   
    TKernel = TransitionKernel(traj_size=T, OutCellID=1000, check_escape=True)

    for i in range(10):
        f = i*0.1
        print('fraction -- ', f)
        TRANS = TKernel.AllTrans(MIDX,frac=f)
       
    
