'''
Created on X-X-X

@author: Ian McDowell and ...
'''

class gibbs_sampler(gene_expression_matrix, gene_names, max_num_iterations, burnIn_phaseI, burnIn_phaseII, s, check_convergence):
    '''
    Explore the posterior distribution of clusterings by sampling the prior based
    on Neal's Algorithm 8 (2000) and the conditional likelihood of a gene being 
    assigned to a particular cluster defined by a Gaussian Process mean function
    and radial basis function covariance.

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
    def __init__(self):
        self.iter_num = 0
        self.num_samples_taken = 0 
        
        # initialize convergence variables
        self.min_sq_dist_counter = 0
        self.post_counter = 0
        self.min_sq_dist = np.inf
        self.current_sq_dist = np.inf
        self.max_post = -np.inf
        self.current_post = -np.inf 
        self.converged = False
        self.converged_by_sq_dist = False
        self.converged_by_likelihood = False
        
        # initialize dictionary to keep track of logpdf of MVN by cluster by gene
        self.last_MVN_by_cluster_by_gene = collections.defaultdict(dict)
        
        # initialize a posterior similarity matrix
        N = gene_expression_matrix.shape[0]
        self.S = np.zeros((N, N))
        
        # initialize a dict liking gene (key) to last cluster assignment (value)
        self.last_cluster = {}
        
        # initialize a list to keep track of log likelihoods
        self.log_likelihoods = []
        
        # initialize arrays to keep track of clusterings
        self.sampled_clusterings = np.array(gene_names)
        self.all_clusterings = np.array(gene_names)
            
    print 'Initializing one-gene clusters...'
    for i, gene_name in enumerate(gene_names):
        self.clusters[m+i] = CLUSTER(members=np.array([i])) # cluster members, D.O.B.
        self.last_cluster[i] = m + i
    
    def get_log_posterior(self):
        '''Get log posterior distribution of the current clustering.'''
        
        prior = np.array([self.clusters[clusterID].size for clusterID in sorted(clusters) if self.clusters[clusterID].size > 0])
        log_prior = np.log( prior / float(prior.sum()) )        
        log_likelihood = np.array([self.clusters[clusterID].model.log_likelihood() for clusterID in sorted(clusters) if self.clusters[clusterID].size > 0 ])
        return ( np.sum(log_prior + log_likelihood) )

    while (not self.converged) and (self.iter_num < max_num_iterations):
        
        self.iter_num += 1
        
        # keep user updated on clustering progress:
        if self.iter_num % 10 == 0:
            print 'Gibbs sampling iteration %s'%(iter_num)
        if self.iter_num == burnIn_phaseI:
            print 'Past burn-in phase I, start optimizing hyperparameters...'
        if self.iter_num == burnIn_phaseII:
            print 'Past burn-in phase II, start taking samples...'
        
        print 'Sizes of clusters =', [c.size for c in self.clusters.values()]
                
        # at every iteration create empty clusters to ensure new mean trajectories
        for i in range(0, m):
            self.clusters[i] = CLUSTER(members=np.array([]))
                
        for gene, gene_name in enumerate(gene_names):
            
            prior = calculate_prior(self.clusters, self.last_cluster, gene)
            lik, LL = calculate_likelihood_MVN_by_dict(self.clusters, self.last_cluster, gene, self.last_MVN_by_cluster_by_gene)
            
            for clusterID, likelihood in zip(sorted(self.clusters), LL):
                if self.clusters[clusterID].model_optimized or self.clusters[clusterID].size <= 1:
                    self.last_MVN_by_cluster_by_gene[clusterID][gene] = likelihood
            
            post = prior * lik
            post = post/sum(post)
            
            cluster_chosen_index = np.where(np.random.multinomial(1, post, size=1).flatten() == 1)[0][0]            
            cluster_chosen = sorted(self.clusters)[cluster_chosen_index]            
            prev_cluster = self.last_cluster[gene]
            
            if (prev_cluster != cluster_chosen): # if a new cluster chosen:
                
                if (clusters[cluster_chosen].size == 0):
                    
                    # create a new cluster
                    cluster_chosen = np.max(self.clusters.keys()) + 1
                    self.clusters[cluster_chosen] = CLUSTER(members=np.array([gene]), iter_num_at_birth=self.iter_num)
                    
                else:
                    
                    self.clusters[cluster_chosen].add_member(gene, self.iter_num)
                
                # remove gene from previous cluster
                self.clusters[prev_cluster].remove_member(gene, self.iter_num)
                
                # if cluster becomes empty, remove
                if (self.clusters[prev_cluster].size == 0):
                    del self.clusters[prev_cluster]
                
            else: # if the same cluster is chosen, then pass
                pass
            
            self.last_cluster[gene] = cluster_chosen
                    
        if (self.iter_num >= burnIn_phaseI):
            
            self.all_clusterings = np.vstack((self.all_clusterings, np.array([self.last_cluster[gene] for gene, gene_name in enumerate(gene_names)])))            
            
            if sample(self.iter_num, s-1):
            
                self.clusters, self.last_MVN_by_cluster_by_gene = update_cluster_attributes(self.iter_num, self.clusters, self.last_MVN_by_cluster_by_gene)
        
        # take a sample from the posterior distribution
        if sample(self.iter_num, s) and (self.iter_num >= burnIn_phaseII):
                        
            self.sampled_clusterings = np.vstack((self.sampled_clusterings, np.array([self.last_cluster[gene] for gene,gene_name in enumerate(gene_names)])))
            
            self.num_samples_taken += 1
            print 'Sample number: %s'%(self.num_samples_taken)
            
            # save log-likelihood of sampled clustering
            self.prev_post = self.current_post
            self.current_post = self.get_log_posterior(clusters)
            self.log_likelihoods.append(self.current_post)
            
            self.S, self.current_sq_dist = update_S_matrix(self.last_cluster, self.S, self.num_samples_taken)
            
            if (check_convergence):
                
                self.prev_sq_dist = self.current_sq_dist
                
                # Check convergence by the squared distance of current pairwise gene-by-gene clustering to mean gene-by-gene clustering
                if self.current_sq_dist <= self.min_sq_dist:
                    with utils.suppress_stdout_stderr():
                        min_sq_dist_clusters = copy.deepcopy(clusters)
                        
                    self.min_sq_dist = self.current_sq_dist
                    self.converged_by_sq_dist = check_GS_convergence_by_sq_dist(self.iter_num, self.prev_sq_dist, self.current_sq_dist)
                    if (self.converged_by_sq_dist):
                        self.min_sq_dist_counter += 1
                    else: # restart counter
                        self.min_sq_dist_counter = 0
                else: # restart counter
                    self.min_sq_dist_counter = 0
                
                # Check convergence by posterior log likelihood
                if self.current_post >= self.max_post:
                    self.converged_by_likelihood = check_GS_convergence_by_likelihood(self.iter_num, self.prev_post, self.current_post)
                    self.max_post = self.current_post
                    if (self.converged_by_likelihood):
                        self.post_counter += 1
                    else: # restart counter
                        self.post_counter = 0
                else: # restart counter
                    self.post_counter = 0
                
                # let metrics of convergence plateau for 10 samples before declaring convergence
                if (self.post_counter >= 10 or self.min_sq_dist >= 10):
                    self.converged = True
                else:
                    self.converged = False
    
    if self.converged:
        if self.converged_by_likelihood:
            print 'Gibbs sampling converged by log-likelihood'
        elif self.converged_by_sq_dist:
            print 'Gibbs sampling converged by least squares distance of gene-by-gene pairwise cluster membership'
    elif (self.iter_num == max_num_iterations):
        print "Maximum number of Gibbs sampling iterations: %s; terminating Gibbs sampling now."%(iter_num)
    
    sampled_clusterings = pd.DataFrame(sampled_clusterings[1:,:], columns=sampled_clusterings[0,:])
    
    return(S, all_clusterings, sampled_clusterings, log_likelihoods, iter_num)