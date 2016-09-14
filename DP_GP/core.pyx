'''
Created on 2016-03-06

@author: Ian McDowell and ...
'''
import cython
from cpython cimport bool 
from sys import float_info
import numpy as np
cimport numpy as np
np.import_array()
import pandas as pd
import collections
import GPy
import scipy
import copy
from DP_GP import utils
from sklearn.preprocessing import scale,StandardScaler

import sys

def squared_dist_two_matrices(S, S_new):   
    '''Compute the squared distance between two numpy arrays/matrices.'''
    diff = S - S_new
    sq_dist = np.sum(np.dot(diff, diff))
    return(sq_dist)

# pre-compute useful value
_LOG_2PI = np.log(2 * np.pi)

#############################################################################################
# 
#    Routine for reading gene expression matrix/matrices
#
#############################################################################################

def read_gene_expression_matrices(gene_expression_matrices, true_times, do_not_mean_center, sigma_n2_shape, sigma_n2_rate):
    '''
    Reads a gene expression matrix or matrices (dataframe or dataframes). 
    If there are multiple matrices given, these are assumed to come from replicates
    and the parameters for the prior of on then noise variance are computed from
    the distribution of the empirical noise variance, assuming an inverse gamma distribution.
    Unless otherwise indicated (i.e. do_not_mean_center=True), expression for each gene
    is mean-centered across the time course.
        
    :param gene_expression_matrices: contains path(s) for gene expression matrix/matrices
    :type gene_expression_matrix: list of string(s)
    :param true_times: if true_times, then use time points in header; else assume equally spaced time points
    :type true_times: bool
    :param do_not_mean_center: if do_not_mean_center, then do not mean-center
    :type do_not_mean_center: bool
    :param sigma_n2_shape: shape parameter for inverse gamma prior for noise variance
    :type sigma_n2_shape: float
    :param sigma_n2_rate: rate parameter for inverse gamma prior for noise variance
    :type sigma_n2_rate: float
    
    :returns: (gene_expression_matrix, gene_names, sigma_n, sigma_n2_shape, sigma_n2_rate, t):
        gene_expression_matrix: posterior similarity matrix
        :type gene_expression_matrix: numpy array of dimension N by P where N=number of genes, P=number of time points
        gene_names: list of gene names
        :type gene_names: list
        sigma_n: within-cluster noise variance
        :type sigma_n: float
        sigma_n2_shape: shape parameter for inverse gamma prior for noise variance, differs from input only if replicates given
        :type sigma_n2_shape: float
        sigma_n2_rate: rate parameter for inverse gamma prior for noise variance, differs from input only if replicates given
        :type sigma_n2_rate: float
        t: time points in time series
        :type t: numpy array of dimension P by 1 where P=number of time points
    
    '''
    
    if len(gene_expression_matrices) > 1: # if replicates provided:
        
        for i, gene_expression_matrix in enumerate(gene_expression_matrices):
            
            gene_expression_matrix = pd.read_csv(gene_expression_matrix, delim_whitespace=True, index_col=0)
            gene_names = list(gene_expression_matrix.index)
            
            if true_times:
                t = np.array(list(gene_expression_matrix.columns)).astype('float')
            else:
                t = np.array(range(gene_expression_matrix.shape[1])).astype('float') # equally spaced time points
            if i != 0:
                gene_expression_array = np.dstack((gene_expression_array, gene_expression_matrix))
            else:
                gene_expression_array = np.array(gene_expression_matrix)
        
        # take gene expression mean across replicates
        gene_expression_matrix = gene_expression_array.mean(axis=2)
        
        # scale each gene's expression across time-series to mean 0, std. dev 1.
        if do_not_mean_center:
            
            gene_expression_matrix = pd.DataFrame(gene_expression_matrix, columns=t)
            gene_expression_matrix.index = gene_names
            gene_expression_array = np.dstack([gene_expression_array[:,:,i] for i in range(gene_expression_array.shape[2])])
            
        else:
            
            # calibrate scaler to mean across replicates
            scaler = StandardScaler().fit(gene_expression_matrix) 
            # standardize mean gene expression
            gene_expression_matrix = scaler.transform(gene_expression_matrix) 
            gene_expression_matrix = pd.DataFrame(gene_expression_matrix, columns=t)
            gene_expression_matrix.index = gene_names
            
            # use scaler to scale replicates separately and create depth-stacked array (depth=reps) of standardized expression
            gene_expression_array = np.dstack([scaler.transform(gene_expression_array[:,:,i]) for i in range(gene_expression_array.shape[2])])
            
        # estimate noise variance function parameters empirically:
        sigma_n2_distro = gene_expression_array.var(axis=2)
        sigma_n2_shape, loc, sigma_n2_scale = scipy.stats.invgamma.fit(sigma_n2_distro, floc=0 )
        sigma_n2_rate = 1.0 / sigma_n2_scale
        # compute sigma_n as median of distribution
        sigma_n = np.sqrt(1 / ((sigma_n2_shape + 1) * sigma_n2_rate))
        
    else: # if there are no replicates:
        
        gene_expression_matrix = pd.read_csv(gene_expression_matrices[0], delim_whitespace=True, index_col=0)
        gene_names = list(gene_expression_matrix.index)
        
        if true_times:
            t = np.array(list(gene_expression_matrix.columns)).astype('float')
        else:
            t = np.array(range(gene_expression_matrix.shape[1])).astype('float') # equally spaced time points
        
        # standardize gene expression
        if not do_not_mean_center:
            gene_expression_matrix = pd.DataFrame(scale(np.array(gene_expression_matrix), axis=1), columns=t)
            gene_expression_matrix.index = gene_names
        else:
            gene_expression_matrix.columns = t
        
        # assign hyperpriors to noise variance:
        sigma_n2_shape = sigma_n2_shape # these value seems to work well in practice,
        sigma_n2_rate = sigma_n2_rate  # without prior, sigma_n may be (over-)optimized to zero.
        # compute sigma_n as median of distribution
        sigma_n = np.sqrt(1 / ((sigma_n2_shape + 1) * sigma_n2_rate))
        
    gene_expression_matrix = np.array(gene_expression_matrix)
    return(gene_expression_matrix, gene_names, sigma_n, sigma_n2_shape, sigma_n2_rate, t)

#############################################################################################
# 
#    Reporting functions
#
#############################################################################################

def save_clusterings(sampled_clusterings, output_path_prefix):
    """Save all sampled clusterings to file."""
    sampled_clusterings.to_csv(output_path_prefix + "_clusterings.txt", sep='\t', index=False)
    
def save_posterior_similarity_matrix(sim_mat, gene_names, output_path_prefix):
    """Save posterior similarity matrix to file."""
    pd.DataFrame(sim_mat, columns=gene_names, index=gene_names).to_csv(output_path_prefix+"_posterior_similarity_matrix.txt", sep='\t')

def save_log_likelihoods(log_likelihoods, output_path_prefix):
    """Save log-likelihoods to file."""
    with open(output_path_prefix + '_log_likelihoods.txt', 'w') as f:
        f.write('\n'.join(["%0.4f"%LL for LL in log_likelihoods]) + '\n')

def save_posterior_similarity_matrix_key(gene_names, output_path_prefix):
    """Save log-likelihoods to file."""
    with open(output_path_prefix + '_posterior_similarity_matrix_heatmap_key.txt', 'w') as f:
        f.write('\n'.join(gene_names) + '\n')

#############################################################################################
# 
#    Define dp_cluster class
#
#############################################################################################

class dp_cluster():
    '''
    dp_cluster object is composed of 0 or more genes and is parameterized by a Gaussian process.
    
    :param members: 0 or more gene indices that belong to cluster
    :type members: list
    :param sigma_n: initial noise variance
    :type sigma_n: float
    :param X: vertically stacked array of time points
    :type X: numpy array of dimension 1 x P, where P = number of time points
    :param iter_num_at_birth: iteration when cluster created
    :type iter_num_at_birth: int
    
    :rtype: dp_cluster object
    
    '''
    def __init__(self, members, sigma_n, X, Y=None, iter_num_at_birth=0):
        
        self.dob = iter_num_at_birth # dob = date of birth, i.e. GS iteration number of creation
        self.members = members # members is a list of gene indices that belong to this cluster.
        self.size = len(self.members) # how many members?
        self.model_optimized = False # a newly created cluster is not optimized
        
        # it may be beneficial to keep track of neg. log likelihood and hyperparameters over iterations
        self.sigma_f_at_iters, self.sigma_n_at_iters, self.l_at_iters, self.NLL_at_iters, self.update_iters = [],[],[],[],[]
        
        # noise variance is initially set to a constant (and possibly estimated) value
        self.sigma_n = sigma_n
        self.X = X
        
        # Define a convariance kernel with a radial basis function and freely allow for a overall slope and bias
        self.kernel = GPy.kern.RBF(input_dim=1, variance = 1., lengthscale = 1.) + \
                      GPy.kern.Linear(input_dim=1, variances=0.001) + \
                      GPy.kern.Bias(input_dim=1, variance=0.001)
        self.K = self.kernel.K(self.X)
                
        if (self.size == 0):
            # for empty clusters, draw a mean vector from the GP prior
            self.Y = np.vstack(np.random.multivariate_normal(np.zeros(self.X.shape[0]), self.K, 1).flatten())
        else: 
            self.Y = Y
        
        self.model = GPy.models.GPRegression(self.X, self.Y, self.kernel)
        self.model.Gaussian_noise = self.sigma_n**2
        self.mean, covK = self.model._raw_predict(X, full_cov=True)
        self.covK = self.K + (self.sigma_n**2) * np.eye(X.shape[0])
        self.mean = np.hstack(self.mean)
        self.update_rank_U_and_log_pdet()
        
    def update_rank_U_and_log_pdet(self):
        ''' 
        [need description]
        '''
        s, u = scipy.linalg.eigh(self.covK, check_finite=False)
        eps = 1E6 * np.finfo('float64').eps * np.max(abs(s))
        d = s[s > eps]
        s_pinv  = np.array([0 if abs(l) <= eps else 1/l for l in s], dtype=float)
        self.rank = len(d)
        self.U = np.multiply(u, np.sqrt(s_pinv))
        self.log_pdet = np.sum(np.log(d))
        
    def add_member(self, new_member, iter_num):
        ''' 
        Add a member to this cluster and increment size.
        '''
        self.members += [new_member]
        self.size += 1
        self.model_optimized = False
        
    def remove_member(self, old_member, iter_num):
        '''
        Remove a member from this cluster and decrement size.
        '''
        self.members = [member for member in self.members if member !=  old_member ]
        self.size -= 1
        self.model_optimized = False
        
    def update_cluster_attributes(self, gene_expression_matrix, sigma_n2_shape, sigma_n2_rate, length_scale_mu, length_scale_sigma, sigma_f_mu, sigma_f_sigma, iter_num, max_iters, optimizer):
        '''
        For all the clusters greater than size 1, update their Gaussian process hyperparameters,
        then return clusters.
        
        :param iter_num: current Gibbs sampling iteration
        :type iter_num: int
        :param clusters: dictionary of dp_cluster objects
        :type clusters: dict
        
        :returns: clusters: dictionary of dp_cluster objects
        :rtype: dict
        
        '''
        
        if (self.size > 0):
            
            if not self.model_optimized:
                
                # update model and associated hyperparameters
                gene_expression_matrix = np.array(gene_expression_matrix)
                self.Y = np.array(np.mat(gene_expression_matrix[self.members,:])).T
                self.model = GPy.models.GPRegression(self.X, self.Y, self.kernel)
                
                # for some reason, must re-set prior on Gaussian noise at every update:
                self.model.Gaussian_noise.set_prior(GPy.priors.InverseGamma(sigma_n2_shape, sigma_n2_rate), warning=False)                
                self.model.sum.rbf.lengthscale.set_prior(GPy.priors.LogGaussian(length_scale_mu, length_scale_sigma), warning=False)
                self.model.sum.rbf.variance.set_prior(GPy.priors.LogGaussian(sigma_f_mu, sigma_f_sigma), warning=False)
                self.model_optimized = True
                self.model.optimize(optimizer, max_iters=max_iters)
                mean, self.covK  = self.model.predict(self.X, full_cov=True, kern=self.model.kern)
                self.sigma_n = np.sqrt(self.model['Gaussian_noise'][0])
                self.mean = np.hstack(mean.mean(axis=1))
                self.K = self.kernel.K(self.X)
                self.update_rank_U_and_log_pdet()
                
            # keep track of neg. log-likelihood and hyperparameters
            self.sigma_n_at_iters.append(self.sigma_n)
#             self.sigma_f_at_iters.append(np.sqrt(self.model.sum['rbf.variance'][0]))
#             self.l_at_iters.append(np.sqrt(self.model.sum['rbf.lengthscale'][0]))
            self.sigma_f_at_iters.append(np.sqrt(self.model.sum.rbf.variance))
            self.l_at_iters.append(np.sqrt(self.model.sum.rbf.lengthscale))
            self.NLL_at_iters.append( - float(self.model.log_likelihood()) )
            self.update_iters.append(iter_num)
            
        else:
            # No need to do anything to empty clusters.
            # New mean trajectory will be drawn at every iteration according to the Gibb's sampling routine
            pass
            
        return self

#############################################################################################
# 
#    Define gibbs_sampler class
#
#############################################################################################
        
cdef class gibbs_sampler(object):
    '''
    Explore the posterior distribution of clusterings by sampling the prior based
    on Neal's Algorithm 8 (2000) and the conditional likelihood of a gene being 
    assigned to a particular cluster defined by a Gaussian Process mean function
    and radial basis function covariance.
    
    :param gene_expression_matrix: expression over timecourse of dimension |genes|x|timepoints|
    :type gene_expression_matrix: numpy array of floats
    :param t: sampled timepoints
    :type t: numpy array of floats
    :param max_num_iterations: maximum number of times gibbs_sampler will loop
    :type max_num_iterations: int
    :param max_iters: maximum number of hyperparameter optimization optimizer iterations
    :type max_iters: int
    :param optimizer: optimizer to use for hyperparameter optimization, see wrapper script usage
    :type optimizer: str
    :param burnIn_phaseI: sampling iteration after which GP hyperparameter optimization takes place
    :type burnIn_phaseI: int
    :param burnIn_phaseII: sampling iteration after which samples are taken
    :type burnIn_phaseII: int
    :param alpha: Dirichlet process concentration parameter
    :type alpha: float
    :param m: number of empty clusters at each ierations
    :type m: int
    :param s: thinning parameter, sample taken at every sth iteration
    :type s: int
    :param s: check convergence? alternately, wait until max_num_iterations obtained
    :type s: bool
    :param sigma_n_init: initial noise standard deviation
    :type sigma_n_init: float
    :param sigma_n2_shape: shape parameter for inverse gamma prior for noise variance
    :type sigma_n2_shape: float
    :param sigma_n2_rate: rate parameter for inverse gamma prior for noise variance
    :type sigma_n2_rate: float
    :param sq_dist_eps: epsilon for similarity matrix squared distance convergence
    :type sq_dist_eps: float
    :param post_eps: epsilon for posterior log likelihood convergence
    :type post_eps: float
    
    :returns: (S, sampled_clusterings, log_likelihoods, iter_num):
        S: posterior similarity matrix
        :type S: numpy array of dimension N by N where N=number of genes
        sampled_clusterings: a pandas dataframe where each column is a gene,
                             each row is a sample, and each record is the cluster
                             assignment for that gene for that sample.
        log_likelihoods: list of log likelihoods of sampled clusterings
        :type log_likelihoods: list
        iter_num: iteration number at termination of Gibbs Sampler
        :type iter_num: int
    
    '''
    
    cdef int iter_num, num_samples_taken, min_sq_dist_counter, post_counter, m, s, burnIn_phaseI, burnIn_phaseII ,max_num_iterations, max_iters, n_genes
    cdef double min_sq_dist, prev_sq_dist, current_sq_dist, max_post, current_post, prev_post, alpha,  sigma_n_init, sigma_n2_shape, sigma_n2_rate, length_scale_mu, length_scale_sigma, sigma_f_mu, sigma_f_sigma, sq_dist_eps, post_eps
    cdef bool converged, converged_by_sq_dist, converged_by_likelihood, check_convergence
    cdef double[:,:] gene_expression_matrix
    cdef double[:] t
    cpdef optimizer, X, last_MVN_by_cluster_by_gene, last_cluster, clusters, S, log_likelihoods, sampled_clusterings, all_clusterings
    
    def __init__(self,
                 double[:,:] gene_expression_matrix,
                 double[:] t,
                 int max_num_iterations, 
                 int max_iters, 
                 optimizer, 
                 int burnIn_phaseI, 
                 int burnIn_phaseII, 
                 double alpha, 
                 int m, 
                 int s, 
                 bool check_convergence, 
                 double sigma_n_init, 
                 double sigma_n2_shape, 
                 double sigma_n2_rate, 
                 double length_scale_mu, 
                 double length_scale_sigma, 
                 double sigma_f_mu, 
                 double sigma_f_sigma, 
                 double sq_dist_eps, 
                 double post_eps):
        
        # hard-coded vars:        
        self.iter_num = 0
        self.num_samples_taken = 0 
        self.min_sq_dist_counter = 0
        self.post_counter = 0
        
        self.min_sq_dist = float_info.max
        self.prev_sq_dist = float_info.max
        self.current_sq_dist = float_info.max
        self.max_post = -float_info.max
        self.current_post = -float_info.max 
        self.prev_post = -float_info.max 
        
        self.converged = False
        self.converged_by_sq_dist = False
        self.converged_by_likelihood = False
        
        # initialize DP parameters
        self.alpha = alpha
        self.m = m
        
        # initialize sampling parameters
        self.s = s
        self.burnIn_phaseI = burnIn_phaseI
        self.burnIn_phaseII = burnIn_phaseII
        self.max_num_iterations = max_num_iterations
        
        # initialize optimization parameters
        self.max_iters = max_iters
        self.optimizer = optimizer
        
        # initialize convergence variables
        self.sq_dist_eps = sq_dist_eps
        self.post_eps = post_eps
        self.check_convergence = check_convergence
        
        # intialize kernel variables
        self.sigma_n_init = sigma_n_init
        self.sigma_n2_shape = sigma_n2_shape
        self.sigma_n2_rate = sigma_n2_rate
        self.length_scale_mu = length_scale_mu
        self.length_scale_sigma = length_scale_sigma
        self.sigma_f_mu = sigma_f_mu
        self.sigma_f_sigma = sigma_f_sigma
        self.t = t
        self.X = np.vstack(t)
        
        # initialize expression
        self.gene_expression_matrix = gene_expression_matrix
        self.n_genes = gene_expression_matrix.shape[0]
        
        # initialize dictionary to keep track of logpdf of MVN by cluster by gene
        self.last_MVN_by_cluster_by_gene = collections.defaultdict(dict)
        
        # initialize a posterior similarity matrix
        N = gene_expression_matrix.shape[0]
        self.S = np.zeros((N, N))
        
        # initialize a dict liking gene (key) to last cluster assignment (value)
        self.last_cluster = {}
        
        # initialize cluster dict in which to keep all DP clusters
        self.clusters = {}
        
        # initialize a list to keep track of log likelihoods
        self.log_likelihoods = []
        
        # initialize arrays to keep track of clusterings
        self.sampled_clusterings = np.arange(self.n_genes)
        self.all_clusterings = np.arange(self.n_genes)
        
    #############################################################################################
        
    def get_log_posterior(self):
        '''Get log posterior distribution of the current clustering.'''
        
        prior = np.array([self.clusters[clusterID].size for clusterID in sorted(self.clusters) if self.clusters[clusterID].size > 0])
        log_prior = np.log( prior / float(prior.sum()) )        
        log_likelihood = np.array([self.clusters[clusterID].model.log_likelihood() for clusterID in sorted(self.clusters) if self.clusters[clusterID].size > 0 ])
        return ( np.sum(log_prior + log_likelihood) )
    
    #############################################################################################
    
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    def calculate_prior(self, int gene):
        '''
        Implementation of Neal's algorithm 8 (DOI:10.1080/10618600.2000.10474879), 
        according to the number of genes in each cluster, returns an array of prior 
        probabilities sorted by cluster name.
        
        :param clusters: dictionary of dp_cluster objects
        :type clusters: dict
        :param last_cluster: dictionary linking gene with the most recent cluster to which gene belonged
        :type last_cluster: dict
        :param gene: gene index
        :type gene: int
        
        :returns: normalized prior probabilities
        :rtype: numpy array of floats
        '''
        
        cdef double[:] prior = np.zeros(len(self.clusters))
        cdef int index
        cdef int clusterID
        
        # Check if the last cluster to which the gene belonged was a singleton.
        singleton = True if self.clusters[self.last_cluster[gene]].size == 1 else False
        
        # If it was a singleton, adjust the number of Chinese Restaurant Process
        # "empty tables" accordingly (e.g. +1).
        if singleton:
            for index, clusterID in enumerate(sorted(self.clusters)):
                if (self.last_cluster[gene] == clusterID): # if gene last belonged to the cluster under consideration
                    prior[index] =  self.alpha / ( self.m + 1 )
                else:
                    if (self.clusters[clusterID].size == 0):
                        prior[index] =  self.alpha / ( self.m + 1 ) 
                    else:
                        prior[index] = float( self.clusters[clusterID].size )
        else:
            for index, clusterID in enumerate(sorted(self.clusters)):
                if (self.last_cluster[gene] == clusterID): # if gene last belonged to the cluster under consideration
                    prior[index] = float( self.clusters[clusterID].size - 1 )
                else:
                    if (self.clusters[clusterID].size == 0):
                        prior[index] =  self.alpha / self.m 
                    else:
                        prior[index] = float( self.clusters[clusterID].size )
        
        prior_normed = prior / np.sum(prior)
        return prior_normed
        
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    def calculate_likelihood_MVN_by_dict(self, int gene):
        '''
        Compute likelihood of gene belonging to each cluster (sorted by cluster name) according
        to the multivariate normal distribution.
        
        :param clusters: dictionary of dp_cluster objects
        :type clusters: dict
        :param last_cluster: dictionary linking gene index with the most recent cluster to which gene belonged
        :type last_cluster: dict
        :param gene: gene index
        :type gene: int
        
        :returns: normalized likelihood probabilities
        :rtype: numpy array of floats
        '''
        
        cdef double[:] lik = np.zeros(len(self.clusters))
        cdef int index
        cdef int clusterID
        
        # Check if the last cluster to which the gene belonged was a singleton.
        singleton = True if self.clusters[self.last_cluster[gene]].size == 1 else False
        
        # expression of gene tested
        expression_vector = self.gene_expression_matrix[gene,:]
        
        for index, clusterID in enumerate(sorted(self.clusters)):
            
            if clusterID in self.last_MVN_by_cluster_by_gene and gene in self.last_MVN_by_cluster_by_gene[clusterID]:
                lik[index] = self.last_MVN_by_cluster_by_gene[clusterID][gene]
            else:
                lik[index] = -0.5 * (self.clusters[clusterID].rank * _LOG_2PI + self.clusters[clusterID].log_pdet + \
                                     np.sum(np.square(np.dot(expression_vector - self.clusters[clusterID].mean, self.clusters[clusterID].U))))
        
        # scale the log-likelihoods down by subtracting (one less than) the largest log-likelihood
        # (which is equivalent to dividing by the largest likelihood), to avoid
        # underflow issues when normalizing to [0-1] interval.
        lik_scaled_down = np.exp(lik - (np.nanmax(lik)-1))
        lik_normed = lik_scaled_down/np.sum(lik_scaled_down)
        return lik_normed, lik
        
    #############################################################################################
    
    def sample(self):
        ''' 
        Check whether a sample should be taken at this Gibbs sampling iteration.
        Take sample if current iteration number is 0 modulo the thinning parameter s.
        
        :param iter_num: current Gibbs sampling iteration
        :type iter_num: int
        
        :rtype bool
        '''
        
        return True if self.iter_num % self.s == 0 else False
    
    #############################################################################################
    
    def update_S_matrix(self):
        '''
        The S matrix, or the posterior similarity matrix, keeps a running average of gene-by-gene
        co-occurence such that S[i,j] = (# samples gene i in same cluster as gene g)/(# samples total).
        Because of the symmetric nature of this matrix, only the lower triangle is updated and maintained.
        
        This matrix may be used as a check for convergence. To this end, return the squared distance
        between the current clustering and the posterior similarity matrix.
        
        :param S: posterior similarity matrix
        :type S: numpy array of dimension N by N where N=number of genes
        :param last_cluster: dictionary linking gene with the most recent cluster to which gene belonged
        :type last_cluster: dict
        :param num_samples_taken: number of samples
        :type num_samples_taken: int
        
        :returns: (S, sq_dist)
            S: posterior similarity matrix
            :type S: numpy array of dimension N by N where N=number of genes
            sq_dist: squared distance between posterior similarity matrix and current clustering
            :type sq_dist: float
        
        '''
        genes1, genes2 = np.tril_indices(len(self.S), k = -1)
        S_new = np.zeros_like(self.S)
        
        for gene1, gene2 in zip(genes1, genes2):
            
            if self.last_cluster[gene1] == self.last_cluster[gene2]:
                S_new[gene1, gene2] += 1.0
            else:
                pass
        
        sq_dist = squared_dist_two_matrices(self.S, S_new)
        
        S = ((self.S * (self.num_samples_taken - 1.0)) + S_new) / float(self.num_samples_taken)
        
        return(S, sq_dist)
    
    #############################################################################################
    
    def check_GS_convergence_by_sq_dist(self):
        ''' 
        Check for GS convergence based on squared distance of current clustering and 
        the posterior similarity matrix.
        
        :param iter_num: current Gibbs sampling iteration
        :type iter_num: int
        :param prev_sq_dist: previous squared distance between point clustering and posterior similarity matrix
        :type prev_sq_dist: float
        :param current_sq_dist: current squared distance between point clustering and posterior similarity matrix
        :type current_sq_dist: int
        
        :rtype bool
        
        '''
        if  self.current_sq_dist == 0 or \
           (abs( (self.prev_sq_dist - self.current_sq_dist) / self.current_sq_dist) <= self.sq_dist_eps \
            and (self.current_sq_dist < self.sq_dist_eps)):
            return True
        else:
            return False
    
    def check_GS_convergence_by_likelihood(self):
        ''' 
        Check for GS convergence based on whether the posterior likelihood of cluster assignment
        changes over consecutive GS samples.
        
        :param iter_num: current Gibbs sampling iteration
        :type iter_num: int
        :param prev_post: previous log-likelihood
        :type prev_post: float
        :param current_post: current log-likelihood
        :type current_post: int
        
        :rtype bool
        
        '''
        if (abs( (self.prev_post - self.current_post) / self.current_post) <= self.post_eps):
            return True
        else:
            return False
    
    #############################################################################################
    
    def sampler(self):
        
        print 'Initializing one-gene clusters...'
        cdef int i, gene
        for i in range(self.n_genes):
            self.clusters[self.m + i] = dp_cluster(members=[i], sigma_n=self.sigma_n_init, X=self.X, Y=np.array(np.mat(self.gene_expression_matrix[i,:])).T, iter_num_at_birth=self.iter_num) # cluster members, D.O.B.
            self.last_cluster[i] = self.m + i
            
        while (not self.converged) and (self.iter_num < self.max_num_iterations):
                        
            self.iter_num += 1
            
            # keep user updated on clustering progress:
            if self.iter_num % 10 == 0:
                print 'Gibbs sampling iteration %s'%(self.iter_num)
            if self.iter_num == self.burnIn_phaseI:
                print 'Past burn-in phase I, start optimizing hyperparameters...'
            if self.iter_num == self.burnIn_phaseII:
                print 'Past burn-in phase II, start taking samples...'
            
            # at every iteration create empty clusters to ensure new mean trajectories
            for i in range(0, self.m):
                self.clusters[i] = dp_cluster(members=[], sigma_n=self.sigma_n_init, X=self.X, iter_num_at_birth=self.iter_num)
                if i in self.last_MVN_by_cluster_by_gene:
                    del self.last_MVN_by_cluster_by_gene[i]
            
            print 'Sizes of clusters =', [c.size for c in self.clusters.values()]
            for i in range(self.n_genes):
                gene = i
                
                prior = self.calculate_prior(gene)
                lik, LL = self.calculate_likelihood_MVN_by_dict(gene)
                
                # lik is array of normalized likelihoods
                # LL is array of log-likelihoods
                for clusterID, likelihood in zip(sorted(self.clusters), LL):
                    self.last_MVN_by_cluster_by_gene[clusterID][gene] = likelihood
                
                post = prior * lik
                post = post/sum(post)
                
                cluster_chosen_index = np.where(np.random.multinomial(1, post, size=1).flatten() == 1)[0][0]
                cluster_chosen = sorted(self.clusters)[cluster_chosen_index]            
                prev_cluster = self.last_cluster[gene]
                
                if (prev_cluster != cluster_chosen): # if a new cluster chosen:
                    
                    if (self.clusters[cluster_chosen].size == 0):
                        
                        # create a new cluster
                        cluster_chosen = max(self.clusters.keys()) + 1
                        self.clusters[cluster_chosen] = dp_cluster(members=[gene], sigma_n=self.sigma_n_init, X=self.X, Y=np.array(np.mat(self.gene_expression_matrix[gene,:])).T, iter_num_at_birth=self.iter_num)
                        
                    else:
                        
                        self.clusters[cluster_chosen].add_member(gene, self.iter_num)
                    
                    # remove gene from previous cluster
                    self.clusters[prev_cluster].remove_member(gene, self.iter_num)
                    
                    # if cluster becomes empty, remove
                    if (self.clusters[prev_cluster].size == 0):
                        
                        del self.clusters[prev_cluster]
                        del self.last_MVN_by_cluster_by_gene[prev_cluster]
                    
                else: # if the same cluster is chosen, then pass
                    pass
                
                self.last_cluster[gene] = cluster_chosen
                
            if (self.iter_num >= self.burnIn_phaseI):
                
                self.all_clusterings = np.vstack((self.all_clusterings, np.array([self.last_cluster[i] for i in range(self.n_genes)])))            
                
                if self.sample():
                    
                    for clusterID, cluster in self.clusters.iteritems():
                        
                        if not cluster.model_optimized and cluster.size > 1:
                            del self.last_MVN_by_cluster_by_gene[clusterID]
                        
                        self.clusters[clusterID] = cluster.update_cluster_attributes(self.gene_expression_matrix, self.sigma_n2_shape, self.sigma_n2_rate, self.length_scale_mu, self.length_scale_sigma, self.sigma_f_mu, self.sigma_f_sigma, self.iter_num, self.max_iters, self.optimizer)
                        
            # take a sample from the posterior distribution
            if self.sample() and (self.iter_num >= self.burnIn_phaseII):
                
                self.sampled_clusterings = np.vstack((self.sampled_clusterings, np.array([self.last_cluster[i] for i in range(self.n_genes)])))   
                
                self.num_samples_taken += 1
                print 'Sample number: %s'%(self.num_samples_taken)
                
                # save log-likelihood of sampled clustering
                self.prev_post = self.current_post
                self.current_post = self.get_log_posterior()
                self.log_likelihoods.append(self.current_post)
                self.S, self.current_sq_dist = self.update_S_matrix()
                
                if (self.check_convergence):
                    
                    self.prev_sq_dist = self.current_sq_dist
                    
                    # Check convergence by the squared distance of current pairwise gene-by-gene clustering to mean gene-by-gene clustering
                    if self.current_sq_dist <= self.min_sq_dist:
                        
                        with utils.suppress_stdout_stderr():
                            min_sq_dist_clusters = copy.deepcopy(self.clusters)
                        
                        self.min_sq_dist = self.current_sq_dist
                        self.converged_by_sq_dist = self.check_GS_convergence_by_sq_dist()
                        
                        if (self.converged_by_sq_dist):
                            self.min_sq_dist_counter += 1
                        else: # restart counter
                            self.min_sq_dist_counter = 0
                            
                    else: # restart counter
                        self.min_sq_dist_counter = 0
                    
                    # Check convergence by posterior log likelihood
                    if self.current_post >= self.max_post:
                        
                        self.converged_by_likelihood = self.check_GS_convergence_by_likelihood()
                        self.max_post = self.current_post
                        
                        if (self.converged_by_likelihood):
                            self.post_counter += 1
                        else: # restart counter
                            self.post_counter = 0
                    
                    else: # restart counter
                        self.post_counter = 0
                    
                    # conservatively, let metrics of convergence plateau for 10 samples before declaring convergence
                    if (self.post_counter >= 10 or self.min_sq_dist_counter >= 10):
                        self.converged = True
                    else:
                        self.converged = False
                                
        if self.converged:
            if self.post_counter >= 10:
                print 'Gibbs sampling converged by log-likelihood'
            if self.min_sq_dist_counter >= 10:
                print 'Gibbs sampling converged by least squares distance of gene-by-gene pairwise cluster membership'
        elif (self.iter_num == self.max_num_iterations):
            print "Maximum number of Gibbs sampling iterations: %s; terminating Gibbs sampling now."%(self.iter_num)
        
        self.sampled_clusterings = pd.DataFrame(self.sampled_clusterings[1:,:], columns=self.sampled_clusterings[0,:])
        self.all_clusterings = pd.DataFrame(self.all_clusterings[1:,:], columns=self.all_clusterings[0,:])
        
        # S is lower triangular, make into full symmetric similarity matrix
        self.S = np.array(self.S + self.S.T + np.eye(len(self.S)))
        
        return(self.S, self.all_clusterings, self.sampled_clusterings, self.log_likelihoods, self.iter_num)
