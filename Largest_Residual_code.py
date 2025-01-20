import numpy as np
import pandas as pd
import optparse,time,pathlib,pickle,random,itertools
from abc import ABCMeta
import random as rd
from backends import BackendMPI as Backend
backend = Backend()# Initialize the MPI backend for parallel computation

    
# Abstract base class for defining inference methods
class Method(metaclass=ABCMeta):
    """
    This abstract base class represents an inference method.
    """

    def __getstate__(self):
        """
        Customize the behavior of pickling the object. Ensures that
        the backend is not pickled (necessary for parallel processing).
        """
        state = self.__dict__.copy()
        del state['backend']  
        return state

# Function to select reads overlapping a specific row index
def select_reads(R, loc, row_ind):
    """
    Select reads and their locations that overlap with the specified row index.

    Args:
        R: List of read sequences.
        loc: List of read start locations .
        row_ind: The row index for which reads are selected.

    Returns:
        A dictionary with selected reads and their corresponding start locations.
    """
    selected_reads = []
    selected_locs = []
    
    for t in range(len(R)):
        reads_t = R[t]
        locs_t = loc[t]
        selected_reads_t = []
        selected_locs_t = []
        
        for i in range(len(reads_t)):
            locind = locs_t[i]
            read_start = locind # Assign the start location
            read_end = locind + len(reads_t[i]) # Calculate the end location of the read based on its length
            # Check if the given row index falls within the range of this read
            if read_start <= row_ind <= read_end:
                selected_reads_t.append(reads_t[i])
                selected_locs_t.append(read_start)
        
        selected_reads.append(selected_reads_t)
        selected_locs.append(selected_locs_t)
    
    return {'Read':selected_reads, 'location':selected_locs}

# Class implementing the EM algorithm for read assignment
class Read_EM(Method):
    def __init__(self, S, weight, AF, data, backend):
        """
        Initialize the Read_EM class with the input parameters.

        Args:
            S: Matrix of current estimated haplotype structure.
            weight: Matrix of current estimated haplotype frequency matrix.
            AF: Allele frequency matrix.
            data: Dictionary reads data containing 'Read'(List of read sequences) and 'location'(List of read start locations) data.
            backend: Parallelization backend.
        """
        
        self.S = S
        self.weight = weight.astype(np.float32)
        # Original read data
        self.data_true = data  
        self.Read_data_true = self.data_true['Read']
        self.loc_data_true = self.data_true['location']
        # Current data being used
        self.data = data 
        self.Read_data = self.data_true['Read']
        self.loc_data = self.data_true['location']
        self.T = len(self.Read_data) # Number of time points
        self.m = self.S.shape[1] # Number of haplotypes
        self.N = self.S.shape[0] # Number of SNPs
        self.AF = AF.astype(np.float32) 
        self.backend = backend
        self.selected_read = {'Read':[], 'location':[]}
        self.row_count = np.zeros(self.N)
        self.log_prob_match = np.log(0.99)  # Log-probability for sequencing error 
        self.log_prob_mismatch = np.log(0.01)  # Log-probability for sequencing error 
        
    def density(self, read, hap_sec): 
        """
        Compute the likelihood f(r_{it}| H(r_{it}) = S_{.j}) in Equation (1).
        """
        check = np.array([int(h == r) for h, r in zip(hap_sec, read)])
        log_prob = np.sum(np.where(check, self.log_prob_match, self.log_prob_mismatch))
        return np.exp(log_prob)
        
    
    def Q_fun(self, i, j ,t): 
        """
        Compute the Q_ijt(S, omega) equation.
        """
        likelihood  = self.density(self.selected_read['Read'][t][i],self.S[self.selected_read['location'][t][i]:self.selected_read['location'][t][i]+len(self.selected_read['Read'][t][i]),j])
        sum_weighted = 0
        for c in range(self.m):
            sum_weighted += self.weight[c, t] * self.density(self.selected_read['Read'][t][i], self.S[self.selected_read['location'][t][i]:self.selected_read['location'][t][i] + len(self.selected_read['Read'][t][i]), c])
        return self.weight[j,t]* likelihood/sum_weighted
        
    
    def Q_fun_w(self, i, j, t):
        """
        Compute the Q_ijt(S, omega) equation for omega update.
        """
        read = self.Read_data[t][i]
        loc = self.loc_data[t][i]
        hap_sec = self.S[loc:loc+len(read), j]
        likelihood = self.density(read, hap_sec)
        sum_weighted = sum(self.weight[c, t] * self.density(read, self.S[loc:loc+len(read), c]) for c in range(self.m))
        return self.weight[j, t] * likelihood / sum_weighted   

    def weight_update(self):
        """
        Update the omega matrix in the M-stepis (Equation (3)).
        """
        curr_weight = np.zeros((self.m, self.T))
        for tt in range(self.T):
            if self.Read_data[tt]:
                K_t = len(self.Read_data[tt])
                for jj in range(self.m):
                    S_sum = 0
                    for ii in range(K_t):
                        S_sum += self.Q_fun_w(ii, jj, tt)
                    curr_weight[jj, tt] = S_sum / K_t
        self.weight = curr_weight
        
    def _inner_calc(self, task):
        """
        Perform the inner calculation for S update in the M-step.
        """
        tt, ii, jj = task
        h_can = self.h_can
        likelihood = self.density(self.selected_read['Read'][tt][ii], h_can[self.selected_read['location'][tt][ii]:self.selected_read['location'][tt][ii] + len(self.selected_read['Read'][tt][ii])])
        return self.Q_fun(ii, jj, tt) * np.log(likelihood)
    
  
    def S_reward(self, jj, h_can):
        """
        Compute the reward for a candidate haplotype configuration in the M-step for S update (Equation (4)).
        """
        self.h_can = h_can
        tasks = []
        for tt in range(self.T):
            for ii in range(len(self.selected_read['Read'][tt])):
                task = (tt, ii, jj)
                tasks.append(task)
        seed_pds = self.backend.parallelize(tasks)
        partial_results = self.backend.map(self._inner_calc, seed_pds)
        logL = np.sum(np.array(self.backend.collect(partial_results)))
        return logL

    def S_update(self, iters):
        """
        Update the S matrix in the M-step.
        """        
        Res = np.sum(np.abs(self.AF - self.S.dot(self.weight)), axis = 1)
        Iter, fails = 0, 0
        while Iter < iters:
            row_ind = np.argsort(Res)[::-1][Iter + fails]
            print('The selected row index is:' + str(row_ind))
            if self.row_count[row_ind]<2:
                self.row_count[row_ind] += 1
                Iter +=1
                self.selected_read= select_reads(self.Read_data, self.loc_data, row_ind)
                row_cand = np.array(list(itertools.product(([0,1]),repeat = self.m)))
                S_new = self.S.copy()
                S_curr = S_new.copy()
                logL = []
                for C in range(2**self.m):
                    S_curr[row_ind,:] = row_cand[C]
                    loglikelihood = 0
                    for jj in range(self.m):
                        loglikelihood = self.S_reward(jj,S_curr[:,jj])+loglikelihood
                    logL.append(loglikelihood)
                cand_ind = np.argmax(logL)
                S_new[row_ind,:] = row_cand[cand_ind]
                self.S = S_new  
            else:
                fails += 1
            
    def M_step(self, Steps, S_step):
        """
        Perform the M-step for the E-M algorithm, updating S and omega iteratively.
        Args:
            Steps: Number of iterations to run the M-step.
            S_step: Number of rows to update in each iteration of S update.
        """
        S_list, W_list = [], []
        for Iter in range(Steps):
            R, Loc = [],[]
            for t in range(self.T):
                random_index = rd.sample(range(len(self.Read_data_true[t])),min(len(self.Read_data_true[t]),500))	 
                r, loc = [], []
                for k in random_index:
            	    r.append(self.Read_data_true[t][k])
            	    loc.append(self.loc_data_true[t][k])
                R.append(r)
                Loc.append(loc)
            self.Read_data = R
            self.loc_data = Loc
            self.data = {'Read':R, 'location':Loc}
            self.weight_update()
            print(self.weight)
            self.S_update(S_step)
            print(self.S)
            curr_S = np.copy(self.S)
            S_list.append(curr_S)
            curr_weight = np.copy(self.weight)
            W_list.append(curr_weight)
        return S_list, W_list
      
    
    
# Parsing input arguments for running the EM algorithm
parser = optparse.OptionParser()
parser.add_option('--AF',action = 'store', dest = 'AF_matrix')
parser.add_option('--Data',action = 'store', dest = 'Data')
parser.add_option('--S_es',action = 'store', dest = 'S_es_matrix')
parser.add_option('--W_es',action = 'store', dest = 'W_es_matrix')
parser.add_option('--M_iter', type='int', dest='M_iter', default=3)
parser.add_option('--S_updates', type='int', dest='S_updates', default=5)

options, args = parser.parse_args()

# Load input matrices and data
S_0 = np.array(pd.read_csv(options.S_es_matrix, sep=" ", header = None),dtype=np.int8)
weight_0 = np.array(pd.read_csv(options.W_es_matrix, sep=" ", header = None), dtype=np.float32)
AF = np.array(pd.read_csv(options.AF_matrix, sep=" ", header = None), dtype=np.float32)
with open(options.Data, 'rb') as f:
    data = pickle.load(f) 
    
# Extract user-defined parameters or use defaults
M_iter = options.M_iter  # Number of M-step iterations
S_updates = options.S_updates  # Number of rows to update in each S update

#########################run the EM algorithm#########################################
def EM_algorithm(S_0, weight_0, AF, data, path, backend, M_iter, S_updates):
    """
    Main function to execute the post-processing E-M algorithm for haplotype reconstruction.
    Args:
        S_0: Initial haplotype structure matrix by HaploSep.
        weight_0: Initial haplotype frequency matrix by HaploSep.
        AF: Allele frequency matrix.
        data: Reads data.
        path: Output path for saving results.
        backend: Parallelization backend.
        M_iter: Number of M-step iterations.
        S_updates: Number of rows to updated in each S update. 
    """
    test = Read_EM(S = S_0, weight = weight_0, AF = AF, data = data, backend=backend)
    start_time = time.time()
    S_list, W_list = test.M_step(M_iter, S_updates)
    end_time = time.time()
    elapsed_time = (end_time - start_time)/3600
    print(elapsed_time)
    np.savez(path + 'LR_result.npz',S=S_list, W=W_list)
   
file_name = str(pathlib.Path(options.S_es_matrix).parent.absolute())+'/'
if __name__ == '__main__':
    EM_algorithm(S_0,weight_0, AF, data, file_name, backend, M_iter, S_updates)

