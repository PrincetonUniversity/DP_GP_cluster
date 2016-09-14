import numpy as np
from DP_GP import utils
from scipy.cluster.hierarchy import fclusterdata
import cython

#############################################################################################

def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)

#############################################################################################

def log_factorial(n):
    return np.log(n + 1)

#############################################################################################

# def compute_mpear(cluster_labels, sim_mat):
#     '''
#     Compute MPEAR (Fritsch and Ickstadt 2009, DOI:10.1214/09-BA414).
#     This function and accessory routines were taken with little 
#     modification from Pyclone (Roth et al. 2014 DOI:10.1038/nmeth.2883).
    
#     :param cluster_labels: cluster labels
#     :type cluster_labels: numpy array of ints
#     :param sim_mat: sim_mat[i,j] = (# samples gene i in cluster with gene j)/(# total samples)
#     :type sim_mat: numpy array of (0-1) floats
    
#     :rtype: float
    
#     '''
#     N = sim_mat.shape[0]
    
#     c = np.exp(log_binomial_coefficient(N, 2))
    
#     num_term_1 = 0
    
#     for j in range(N):
#         for i in range(j):
#             if cluster_labels[i] == cluster_labels[j]:
#                 num_term_1 += sim_mat[i][j]
    
#     num_term_2 = 0
    
#     for j in range(N):
#         for i in range(j):
#             if cluster_labels[i] == cluster_labels[j]:
#                 num_term_2 += sim_mat[:j - 1, j].sum()
    
#     num_term_2 /= c
    
#     den_term_1 = 0
    
#     for j in range(N):
#         for i in range(j):
#             den_term_1 += sim_mat[i][j]
            
#             if cluster_labels[i] == cluster_labels[j]:
#                 den_term_1 += 1
    
#     den_term_1 /= 2
    
#     num = num_term_1 - num_term_2
#     den = den_term_1 - num_term_2
    
#     return num / den

#############################################################################################

cdef compute_mpear(int[:] cluster_labels, double[:,:] sim_mat):
    '''
    Compute MPEAR (Fritsch and Ickstadt 2009, DOI:10.1214/09-BA414).
    This function and accessory routines were taken with little 
    modification from Pyclone (Roth et al. 2014 DOI:10.1038/nmeth.2883).
    
    :param cluster_labels: cluster labels
    :type cluster_labels: numpy array of ints
    :param sim_mat: sim_mat[i,j] = (# samples gene i in cluster with gene j)/(# total samples)
    :type sim_mat: numpy array of (0-1) floats
    
    :rtype: float
    
    '''
    cdef int N = sim_mat.shape[0]
    cdef double c = np.exp(log_binomial_coefficient(N, 2))
    cdef double num_term_1 = 0
    cdef double num_term_2 = 0
    cdef double den_term_1 = 0
    
    for j in range(N):
        for i in range(j):
            den_term_1 += sim_mat[i][j]
            if cluster_labels[i] == cluster_labels[j]:
                num_term_1 += sim_mat[i][j]
                num_term_2 += sim_mat[:j - 1, j].sum()
                den_term_1 += 1
    
    num_term_2 /= c
    den_term_1 /= 2
    
    num = num_term_1 - num_term_2
    den = den_term_1 - num_term_2
    
    return num / den

#############################################################################################

def relabel_clustering(cluster_labels):
    '''
    Given some cluster labels, relabel (equivalently) so that labels start at 1.
    '''
    clust_dict = {}
    new_label = 1
    new_labels = []
    for label in list(cluster_labels):
        if label not in clust_dict:
            new_labels.append(new_label)
            clust_dict[label] = new_label
            new_label += 1
        elif label in clust_dict:
            new_labels.append(clust_dict[label])
    
    return new_labels

#############################################################################################

def compute_sq_dist(S, S_new):   
    '''
    Compute the squared distance between two matrices.    
    '''
    diff = S - S_new
    sq_dist = np.sum(np.dot(diff, diff))
    return(sq_dist)


#############################################################################################

def best_clustering_by_mpear(clusterings, sim_mat):
    """
    Find the optimal clustering according to the MPEAR criterion.
    
    :param clusterings: clusterings[i,j] is the cluster to which gene j belongs at sample i
    :type clusterings: numpy array of ints
    :param sim_mat: sim_mat[i,j] = (# samples gene i in cluster with gene j)/(# total samples)
    :type sim_mat: numpy array of (0-1) floats
    
    :rtype: numpy array of best cluster labels
    
    """
    max_pear = 0
    
    for i in range(len(clusterings)):
        
        pear = compute_mpear(clusterings[i,], sim_mat)
        
        if pear > max_pear:
            
            max_pear = pear
            best_cluster_labels = clusterings[i]
    
    new_labels = relabel_clustering(best_cluster_labels)
    return new_labels

#############################################################################################

def best_clustering_by_log_likelihood(clusterings, log_post_list):
    """
    Find the optimal clustering according to log posterior likelihood
    (i.e. maximum a posteriori clustering).
    
    :param clusterings: clusterings[i,j] is the cluster to which gene j belongs at sample i
    :type clusterings: numpy array of ints
    :param log_post_list: list of log posterior likelihood over the course of Gibbs sampling
    :type log_post_list: list of floats
    
    :rtype: numpy array of best cluster labels
    
    """
    max_post = -np.inf
    
    for i, post in zip(range(len(clusterings)), log_post_list):
                
        if post > max_post:
            
            max_post = post
            best_cluster_labels = clusterings[i]
    
    new_labels = relabel_clustering(best_cluster_labels)
    return new_labels

#############################################################################################

def best_clustering_by_sq_dist(clusterings, sim_mat):
    """
    Find the optimal clustering according to the Dahl 2006 least-squares criterion
    ("Model-based clustering for expression data via a Dirichlet process mixture model").
    
    :param clusterings: clusterings[i,j] is the cluster to which gene j belongs at sample i
    :type clusterings: numpy array of ints
    :param sim_mat: sim_mat[i,j] = (# samples gene i in cluster with gene j)/(# total samples)
    :type sim_mat: numpy array of (0-1) floats
    
    :rtype: numpy array of best cluster labels
    
    """
    
    min_dist = np.inf
    
    for i in range(len(clusterings)):
        clustering = clusterings[i,:]
        S = np.zeros((len(clustering), len(clustering)))
        for j in range(len(clustering)):
            for k in range(len(clustering)):
                if clustering[j] == clustering[k]:
                    S[j,k] = 1
                else:
                    S[j,k] = 0
        
        dist = compute_sq_dist(S, sim_mat)
        
        if dist < min_dist:
            
            min_dist = dist
            best_cluster_labels = clusterings[i]
    
    new_labels = relabel_clustering(best_cluster_labels)
    return new_labels

def best_clustering_by_h_clust(clusterings, method):
    """
    Find the optimal clustering by hierarchical clustering. For more details, see
    description of scipy.cluster.hierarchy.fclusterdata.
    
    :param clusterings: clusterings[i,j] is the cluster to which gene j belongs at sample i
    :type clusterings: numpy array of ints
    :param sim_mat: sim_mat[i,j] = (# samples gene i in cluster with gene j)/(# total samples)
    :type sim_mat: numpy array of (0-1) floats
    
    :rtype: numpy array of best cluster labels
    
    """
    best_cluster_labels = fclusterdata(clusterings, 0.99, method=method, metric='hamming')
    new_labels = relabel_clustering(best_cluster_labels)
    return new_labels

#############################################################################################

def save_cluster_membership_information(optimal_cluster_labels, output):
    """
    Save cluster membership information in the form:
    cluster<tab>gene
    (e.g.) 1<tab>gene1
              .
              .
              .
    """
    
    handle = open(output, "w")
    handle.write('cluster\tgene\n')
    
    for cluster in sorted(optimal_cluster_labels):
        genes = optimal_cluster_labels[cluster]
        genes = utils.sorted_nicely(genes)
        for gene in genes:
            handle.write('%s\t%s\n'%(cluster, gene))
    
    handle.close()
