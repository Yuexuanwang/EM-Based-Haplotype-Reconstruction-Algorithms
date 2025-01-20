import numpy as np
import pandas as pd
import optparse,time,pathlib,pickle,random,itertools,re
from abc import ABCMeta
import random as rd
import networkx as nx
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


def coloring_algorithm(G):
    """
    The graph coloring algorithm to group the read data by the potential haplotype.
    
    Args:
        G:Empty color map.

    Returns:
        The colored map of each read data.
    """
    # Step 1: Set U=1 and C_i=0 for all i
    U = 1
    color_map = {node: 0 for node in G.nodes()}
    uncolored_nodes = set(G.nodes())

    while uncolored_nodes:
        # Step 2: Assign C_i=1 for some randomly chosen i from the set of uncolored nodes
        i = random.choice(list(uncolored_nodes))
        color_map[i] = 1
        uncolored_nodes.remove(i)

        # Step 5: Repeat from step 3 until all connecting nodes of V_i are colored
        while True:
            neighbors_to_color = [j for j in G.neighbors(i) if color_map[j] == 0]
            if not neighbors_to_color:
                break

            # Step 3: Find the j that gives argmax_j |nC_j|
            def count_neighbor_colors(j):
                neighbor_colors = {color_map[n] for n in G.neighbors(j) if color_map[n] != 0}
                return len(neighbor_colors)

            j = max(neighbors_to_color, key=count_neighbor_colors)

            # Calculate the set of colors of node connecting V_j
            nC_j = {color_map[n] for n in G.neighbors(j) if color_map[n] != 0}

            # Step 4: If |nC_j| = U, set C_j = U + 1, U = U + 1. Else C_j = min{C not in |nC_j|}
            if len(nC_j) == U:
                color_map[j] = U + 1
                U += 1
            else:
                color_map[j] = min(set(range(1, U + 1)) - nC_j)

            uncolored_nodes.remove(j)

    # Step 6: All nodes are colored, return the color_map
    return color_map

def read_confliction(ReadA, ReadB, start1, start2):
    """
    Detect if two reads have confliction.
    
    Args:
        ReadA: The first read sequence.
        ReadB: The second read sequence.
        start1: The start location of first read.
        start2: The start location of second read.
        

    Returns:
        If two reads have confliction.
    """
    L1 = len(ReadA)
    L2 = len(ReadB)
    end1 = start1 + L1 
    end2 = start2 + L2 
    # Find the overlap region
    start_overlap = max(start1, start2)
    end_overlap = min(end1, end2)

    # Check if there is an overlap
    if start_overlap < end_overlap:
        overlap_index_A = start_overlap - start1
        overlap_index_B = start_overlap - start2
        overlap_length = end_overlap - start_overlap
        overlap_A = ReadA[overlap_index_A:overlap_index_A + overlap_length]
        overlap_B = ReadB[overlap_index_B:overlap_index_B + overlap_length]
        # Check for differences in the overlap
        for a, b in zip(overlap_A, overlap_B):
            if a != b:
                return True  
        return False  
    else:
        return False
    
def merge_matrices(S1, S2):
    """
    The function to merge old and new pool of candidate haplotypes.
    
    Args:
        S1: New pool.
        S2: Old pool.

    Returns:
        Merged candidate pool.
    """
    if S1.shape == (0,0):
        S1 = S2
    else:
        new_columns = [col for col in S2.T if not any(np.array_equal(col, S1_col) for S1_col in S1.T)]
        if len(new_columns)>0:
            S1 = np.hstack((S1, np.array(new_columns).T))
    return S1

def poolGen(R, Loc, N): 
    """
    Generate pool of candidate haplotypes by coloring algorithm.
    
    Args:
        R: List of read sequences.
        loc: List of read start locations .
        N: Number of SNPs.

    Returns:
        A pool of candidate haplotypes for S update.
    """
    G = nx.Graph()
    for index_T in range(len(R)): #By networkx define all nodes and edges
        for index_N in range(len(R[0])):
            G.add_node(str(index_T)+','+str(index_N), read =  R[index_T][index_N], pos = Loc[index_T][index_N])
            for index_t in range(index_T+1):
                for index_n in range( (index_t == index_T)*(index_N) + (index_t != index_T)*(len(R[0])) ): 
                    Loc_list = [G.nodes[str(index_T)+','+str(index_N)]['pos'], G.nodes[str(index_t)+','+str(index_n)]['pos']]
                    R_list = [G.nodes[str(index_T)+','+str(index_N)]['read'], G.nodes[str(index_t)+','+str(index_n)]['read']]
                    if read_confliction(R_list[0],R_list[1],Loc_list[0],Loc_list[1]):
                        G.add_edge(str(index_T)+','+str(index_N), str(index_t)+','+str(index_n))

    network = coloring_algorithm(G)   

    pool = - np.ones((N, max(network.values()))) #assign the reads into pool by their color
    #handling missing value
    r_ind = []
    for i in range(max(network.values())):
        # ind = np.where(np.array(list(network.values()),dtype=np.float64) -1 == i)[0]
        ind = np.where(np.array(list(network.values()),dtype=np.float64) == i+1)[0]
        for k in range (len(ind)):
            node = re.findall(r'\d+', list(network.keys())[ind[k]])
            indT = int(node[0])
            indN = int(node[1])
            pool[Loc[indT][indN]:Loc[indT][indN]+len(R[indT][indN]), i] = R[indT][indN]
        zero_ind = np.where(pool[:,i] == -1)[0]
        
        if len(zero_ind)>0:
            pool_cand = pool[:,i].copy()
            pool_cand[zero_ind] = np.random.binomial(1,0.5,len(zero_ind))
            pool[:,i] = pool_cand
            
    return pool


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
        self.row_count = np.zeros(self.N)
        self.log_prob_match = np.log(0.99)  # Log-probability for sequencing error 
        self.log_prob_mismatch = np.log(0.01)  # Log-probability for sequencing error 
        self.pool = merge_matrices(poolGen(self.Read_data, self.loc_data, self.N), self.S)
        
    def density(self, read, hap_sec): 
        """
        Compute the likelihood f(r_{it} | H(r_{it}) = S_{.j} in Equation (1).
        """
        check = np.array([int(h == r) for h, r in zip(hap_sec, read)])
        log_prob = np.sum(np.where(check, self.log_prob_match, self.log_prob_mismatch))
        return np.exp(log_prob)
        
    
    def Q_fun(self, i, j ,t): 
        """
        Compute the Q_ijt(S, omega) equation.
        """
        likelihood  = self.density(self.data['Read'][t][i],self.S[self.data['location'][t][i]:self.data['location'][t][i]+len(self.data['Read'][t][i]),j])
        sum_weighted = 0
        for c in range(self.m):
            sum_weighted += self.weight[c, t] * self.density(self.data['Read'][t][i], self.S[self.data['location'][t][i]:self.data['location'][t][i] + len(self.data['Read'][t][i]), c])
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
        likelihood = self.density(self.data['Read'][tt][ii], h_can[self.data['location'][tt][ii]:self.data['location'][tt][ii] + len(self.data['Read'][tt][ii])])
        return self.Q_fun(ii, jj, tt) * np.log(likelihood)
    
  
    def S_reward(self, jj, h_can):
        """
        Compute the reward for a candidate haplotype configuration in the M-step for S update (Equation (4)).
        """
        self.h_can = h_can
        tasks = []
        for tt in range(self.T):
            for ii in range(len(self.data['Read'][tt])):
                task = (tt, ii, jj)
                tasks.append(task)
        seed_pds = self.backend.parallelize(tasks)
        partial_results = self.backend.map(self._inner_calc, seed_pds)
        logL = np.sum(np.array(self.backend.collect(partial_results)))
        return logL

             
            
    def S_update(self): 
        """
        Update the S matrix in the M-step.
        """   
        new_pool = poolGen(self.Read_data, self.loc_data, self.N)
        merged_pool = merge_matrices(new_pool,self.pool)
        print(new_pool.shape, merged_pool.shape)
        curr_S = np.zeros([self.N, self.m])
        for jj in range(self.m):
            Ll = []
            for Iter in range(merged_pool.shape[1]):
                logL = self.S_reward(jj, merged_pool[:,Iter])
                Ll.append(logL)
            best_pos = np.argmax(Ll)
            curr_S[:,jj] = merged_pool[:, best_pos]
        self.S = curr_S
        if len(set(map(tuple, curr_S.T)) - set(map(tuple, self.pool.T)))!=0:
            self.pool = merged_pool
        print(self.pool.shape)
        
           
    def M_step(self, Steps): 
        """
        Perform the M-step for the E-M algorithm, updating S and omega iteratively.
        Args:
            Steps: Number of iterations to run the M-step.
        """
        S_list, W_list = [],[]
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
            self.S_update()
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

options, args = parser.parse_args()

# Load input matrices and data
S_0 = np.array(pd.read_csv(options.S_es_matrix, sep=" ", header = None),dtype=np.int8)
weight_0 = np.array(pd.read_csv(options.W_es_matrix, sep=" ", header = None), dtype=np.float32)
AF = np.array(pd.read_csv(options.AF_matrix, sep=" ", header = None), dtype=np.float32)
with open(options.Data, 'rb') as f:
    data = pickle.load(f) 
    
# Extract user-defined parameters or use defaults
M_iter = options.M_iter  # Number of M-step iterations

#########################run the EM algorithm#########################################
def EM_algorithm(S_0, weight_0, AF, data, path, backend, M_iter):
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
    """
    test = Read_EM(S = S_0, weight = weight_0, AF = AF, data = data, backend=backend)
    start_time = time.time()
    S_list, W_list = test.M_step(M_iter)
    end_time = time.time()
    elapsed_time = (end_time - start_time)/3600
    print(elapsed_time)
    np.savez(path + 'CB_result.npz',S=S_list, W=W_list)
   
file_name = str(pathlib.Path(options.S_es_matrix).parent.absolute())+'/'
if __name__ == '__main__':
    EM_algorithm(S_0,weight_0, AF, data, file_name, backend, M_iter)

