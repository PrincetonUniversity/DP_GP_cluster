#!/usr/bin/env python
# -*- coding: utf-8 -*-
#############################################################################################
#
#    Create gene-by-gene posterior similarity matrix by clustering gene expression profiles 
#    over time course using Dirichlet Process/Gaussian Process model and select an optimal
#    clustering from one of a range of criteria.
#
#    DP_GP_cluster_post_gibbs_sampling.py 
#
#    Authors: Ian McDowell, Dinesh Manandhar
#    Last Updated: 04/13/2016
#
#    Requires the following Python packages: 
#    pandas, numpy, scipy (>= 0.14)
#############################################################################################
# import dependendencies:
import matplotlib
matplotlib.use('Agg')
font = {'size'   : 8}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

from DP_GP import utils
from DP_GP import cluster_tools
from DP_GP import core
import pandas as pd
import numpy as np
import scipy
import argparse
import collections
from sklearn.preprocessing import scale

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, \
description="""

DP_GP_cluster_post_gibbs_sampling.py takes a similarity matrix or
clustering array as input and returns an optimal clustering according
to one of a number of available criteria.

""")

##############################################################################
#
#  Parse arguments
#
##############################################################################

parser.add_argument("-i", "--input", dest="gene_expression_matrix", action="store", \
                  help="""required, e.g. /path/to/gene_expression_matrix.txt

or /path/to/gene_expression_matrix.rep1.txt,/path/to/gene_expression_matrix.rep2.txt,etc.
if there are replicates

where the format of the gene_expression_matrix.txt is:

gene    1     2    3    ...    p
gene_1  10    20   5    ...    8
gene_2  3     2    50   ...    8
gene_3  18    100  10   ...    22
...
gene_n  45    22   15   ...    60

Note that the first row is a header and the first 
column contains all gene names. Entries are delimited
by whitespace (space or tab), and for this reason, 
do not include spaces in gene names or time-point names.

""")
parser.add_argument("-s", "--sim_mat", dest="sim_mat", action="store", default=None, \
                  help="""optional, e.g. /path/to/similarity_matrix.txt

where the format of the similarity_matrix.txt is:

        gene_0 gene_1 gene_2    gene_n
gene_0  1.0    0.89   0.12  ... 0.0
gene_1  0.89   1.0    0.2   ... 0.01
gene_2  0.12   0.2    1.0   ... 0.7
...     ...    ...    ...   ... ...
gene_n  0.0    0.01   0.7   ... 1.0

Note that the first row is a header and the first 
column is an index, both of which contains an identical
list of genes.  In each cell, S[i,j], is the fraction of 
samples that gene_i was in the same cluster as gene_j.
Thus, all entries are in the unit interval [0,1].
Entries are delimited by whitespace (space or tab), 
and for this reason, do not include spaces in gene names.

Required for the following criteria:
h_clust_avg
h_clust_comp
least_squares

""")
parser.add_argument("--clusterings", dest="clusterings", action="store",  default=None, \
                  help="""optional, e.g. /path/to/clusterings.txt

where the format of the clusterings.txt is:

gene_0 gene_1 gene_2    gene_n
1      1      2     ... 33
10     10     2     ... 33
13     13     13    ... 33
...    ...    ...   ... ... 
49     17     17    ... 100

Note that the first row is a header that lists all genes.
Each row is a different sample from the posterior distribution
of clusterings. Integer values denote cluster membership.
Cluster numbers need not correspond across rows/samples, and
only necessarily apply within row/sample.

Required for the following criteria:
MPEAR 
MAP
least_squares

""")
parser.add_argument("--true_times", action='store_true', \
                  help=("""optional, [default=False]   
Set this flag if the header contains true time values (e.g. 0, 0.5, 4, 8,...)
and it is desired that the covariance kernel recognizes the true
time spacing between sampling points, which need not be constant.
Otherwise, it is assumed that the sampling times are equally spaced, 
or in other words, that the rate of change in expression is equivalent
between all neighboring time points.

"""))
parser.add_argument("-l", "--log_likelihoods", dest="log_likelihoods", action="store",  default=None, \
                  help=("""optional, e.g. /path/to/log_likelihoods.txt

where the format of the log_likelihoods.txt is:

-10200.322
-9987.452
-12291.992
...
-10002.403

Each row corresponds to the posterior log-likelihood and also
corresponds to each row in clusterings.txt. 

Required for the following criterion:
MAP

"""))
parser.add_argument("-c", "--criterion", dest="criterion", type=str, default='MPEAR', \
                  help=("""optional, [default=MPEAR]

Specify the criterion by which you would like to select the optimal clustering
among "MPEAR", "MAP", "least_squares", "h_clust_avg", and "h_clust_comp" where:

MPEAR = Posterior expected adjusted Rand (see Fritsch and Ickstadt 2009, DOI:10.1214/09-BA414) 
MAP = maximum a posteriori
h_clust_avg = hierarchical clustering by average linkage
h_clust_comp = hierarchical clustering by complete linkage
least_squares = minimize squared distance between clustering and posterior similarity matrix 
(see Dahl 2006, "Model-Based Clustering...")

"""))
parser.add_argument("--plot", action='store_true', \
                  help="""optional, [default=do not plot anything, if indicated, then plot

""")
parser.add_argument("-o", "--output", dest="output", action="store", required=True, \
                  help="required, e.g. /path/to/optimal_clustering.txt")
parser.add_argument("-p", "--plot_types", dest="plot_types", type=str, default='pdf', \
                  help="""optional, [default=pdf] plot type, e.g. pdf.
If multiple plot types are desired then separate by commas, e.g. pdf,png
and one of each kind specified will be generated.

""")
parser.add_argument("--output_path_prefix", dest="output_path_prefix", action="store", \
                  help="""required, e.g. /path/to/my_gene_clustering_results
 
Output files generated will be:

Heatmap of posterior similarity matrix, to which complete-linkage hierarchical clustering
has been applied for the purposes of attractiveness and intelligibility:
/path/to/my_gene_clustering_results_posterior_similarity_matrix_heatmap.pdf

Key to ordering of genes in heatmap of posterior similarity matrix:
/path/to/my_gene_clustering_results_posterior_similarity_matrix_heatmap_key.txt

Cluster assignment over Gibbs Sampling iterations (after burn-in and thinning):
/path/to/my_gene_clustering_results_posterior_similarity_matrix_heatmap.pdf

Cluster sizes over course of Gibbs Sampling (including burn-in phase I):
/path/to/my_gene_clustering_results_cluster_sizes.pdf

tab-delimited file listing (col 1) optimal cluster number (col 2) gene name
/path/to/output_path_prefix_optimal_clustering.txt

Plot of gene expression over the time-course, for all optimal clusters
with six panels/clusters per figure:
/path/to/output_path_prefix_gene_expression_fig_1.pdf
.
.
.
/path/to/output_path_prefix_gene_expression_fig_N.pdf

""")

parser.add_argument("--optimizer", dest="optimizer", type=str, default='lbfgsb', \
                  help="""optional, [default=lbfgsb]
Specify the optimization technique used to update GP hyperparameters

lbfgsb = L-BFGS-B 
fmin_tnc = truncated Newton algorithm 
simplex = Nelder-Mead simplex 
scg = stochastic conjugate gradient 

""")
parser.add_argument("-t", "--time_unit", dest="time_unit", type=str, default='', \
                  help="""optional, [default=None] time unit, used for plotting purposes.
 
""")
parser.add_argument("--sigma_n2_shape", dest="sigma_n2_shape", type=float, default=12.,
                  help="""optional, [default=12 or estimated from replicates]

""")
parser.add_argument("--sigma_n2_rate", dest="sigma_n2_rate", type=float, default=2.,
                  help="""optional, [default=2 or estimated from replicates]

""")
parser.add_argument("--length_scale_mu", dest="length_scale_mu", type=float, default=0.,
                  help="""optional, Log normal mean (mu, according to Bishop 2006 conventions) 
for length scale [default=0]

""")
parser.add_argument("--length_scale_sigma", dest="length_scale_sigma", type=float, default=1.,
                  help="""optional, Log normal standard deviation (sigma, according to Bishop 2006 conventions) 
for length scale [default=1]

""")
parser.add_argument("--sigma_f_mu", dest="sigma_f_mu", type=float, default=0.,
                  help="""optional, Log normal mean (mu, according to Bishop 2006 conventions) 
for signal variance [default=0]

""")
parser.add_argument("--sigma_f_sigma", dest="sigma_f_sigma", type=float, default=1.,
                  help="""optional, Log normal standard deviation (sigma, according to Bishop 2006 conventions) 
for signal variance [default=1]

""")
parser.add_argument("--do_not_mean_center", action='store_true', dest="do_not_mean_center", \
                  help="""optional, [default=False]   
Set this flag if you desire the gene expression data to be clustered
without mean-centering (left untransformed).

""")

parser.add_argument('--version', action='version', version='DP_GP_cluster_post_gibbs_sampling.py v.0.1')

args = parser.parse_args()

criterion = args.criterion
clusterings = args.clusterings
sim_mat = args.sim_mat
log_likelihoods = args.log_likelihoods
output = args.output
true_times = args.true_times
time_unit = args.time_unit
output_path_prefix = args.output_path_prefix
plot_types = args.plot_types.split(',')

length_scale_mu, length_scale_sigma, sigma_f_mu, sigma_f_sigma = \
args.length_scale_mu, args.length_scale_sigma, args.sigma_f_mu, args.sigma_f_sigma

if criterion == 'MPEAR':
    if not clusterings or not sim_mat:
        print "ERROR: if criterion = MPEAR, must provide both clusterings and similarity_matrix"
        exit()
elif criterion == 'MAP':
    if not clusterings or not log_likelihoods:
        print "ERROR: if criterion = MAP, must provide both clusterings and log_likelihoods"
        exit()
elif criterion == 'h_clust_avg':
    if not sim_mat:
        print "ERROR: if criterion = h_clust_avg, must provide similarity_matrix"
        exit()
elif criterion == 'h_clust_comp':
    if not sim_mat:
        print "ERROR: if criterion = h_clust_comp, must provide similarity_matrix"
        exit()
elif criterion == 'least_squares':
    if not clusterings or not sim_mat:
        print "ERROR: if criterion = least_squares, must provide both clusterings and similarity_matrix"
        exit()
else:
    print """ERROR: incorrect criterion. Please choose from among the following options:
MPEAR, MAP, least_squares, h_clust_avg, h_clust_comp"""
    exit()

if clusterings:
    clusterings = pd.read_csv(clusterings, delim_whitespace=True)
    gene_names = list(clusterings.columns)

if sim_mat:
    sim_mat = pd.read_csv(sim_mat, delim_whitespace=True, index_col=0)
    gene_names = list(sim_mat.columns)
else:
    sim_mat = []

if log_likelihoods:
    with open(log_likelihoods, 'r') as f:
        log_likelihoods = [float(line.strip()) for line in f]

####
# Cluster
####
# Find an optimal "clusters" list, sorted by gene in order of "gene_names" object

if criterion == 'MPEAR':
    optimal_clusters = cluster_tools.best_clustering_by_mpear(np.array(clusterings), np.array(sim_mat))
elif criterion == 'MAP':
    optimal_clusters = cluster_tools.best_clustering_by_log_likelihood(np.array(clusterings), log_likelihoods)
elif criterion == 'least_squares':
    optimal_clusters = cluster_tools.best_clustering_by_sq_dist(np.array(clusterings), np.array(sim_mat))
elif criterion == 'h_clust_avg':
    optimal_clusters = cluster_tools.best_clustering_by_h_clust(np.array(sim_mat), 'average')
elif criterion == 'h_clust_comp':
    optimal_clusters = cluster_tools.best_clustering_by_h_clust(np.array(sim_mat), 'complete')

optimal_cluster_labels = collections.defaultdict(list)
optimal_cluster_labels_original_gene_names = collections.defaultdict(list)
for gene, (gene_name, cluster) in enumerate(zip(gene_names, optimal_clusters)):
    optimal_cluster_labels[cluster].append(gene)
    optimal_cluster_labels_original_gene_names[cluster].append(gene_name)

    
#############################################################################################
#
#    Import gene expression (and estimate noise variance from data if replicates available)
#
#############################################################################################
if args.gene_expression_matrix:
    gene_expression_matrices = args.gene_expression_matrix.split(',')
    
    def read_gene_expression_matrices_return_array_and_sigma_n(gene_expression_matrices):
        
        if len(gene_expression_matrices) > 1: # if there are replicates:
            
            for i, gene_expression_matrix in enumerate(gene_expression_matrices):
                
                gene_expression_matrix = pd.read_csv(gene_expression_matrix, delim_whitespace=True, index_col=0)
                gene_names = list(gene_expression_matrix.index)
                if true_times:
                    t = np.array(list(gene_expression_matrix.columns)).astype('float')
                else:
                    t = np.array(range(gene_expression_matrix.shape[1])).astype('float') # equally spaced time points
                try:
                    gene_expression_array = np.dstack((gene_expression_array, gene_expression_matrix))
                except NameError:
                    gene_expression_array = np.array(gene_expression_matrix)
                
            # take gene expression mean across replicates
            gene_expression_matrix = gene_expression_array.mean(axis=2)
            
            # scale each gene's expression across time-series to mean 0, std. dev 1.
            if args.do_not_mean_center:
                gene_expression_matrix = pd.DataFrame(gene_expression_matrix, columns=t)
                gene_expression_matrix.index = gene_names
                gene_expression_array = np.dstack([gene_expression_array[:,:,i] for i in range(gene_expression_array.shape[2])])
            else:
                scaler = StandardScaler().fit(gene_expression_matrix) 
                gene_expression_matrix = scaler.transform(gene_expression_matrix) 
                gene_expression_matrix = pd.DataFrame(gene_expression_matrix, columns=t)
                gene_expression_matrix.index = gene_names
                gene_expression_array = np.dstack([scaler.transform(gene_expression_array[:,:,i]) for i in range(gene_expression_array.shape[2])])
            
            # estimate noise variance empirically:
            sigma_n2_distro = gene_expression_array.var(axis=2)
            sigma_n2_shape, loc, sigma_n2_scale = scipy.stats.invgamma.fit(sigma_n2_distro, floc=0 )
            sigma_n2_rate = 1.0 / sigma_n2_scale
            sigma_n = np.sqrt(1 / ((sigma_n2_shape + 1) * sigma_n2_rate))
            
        else: # if there are no replicates:
            
            gene_expression_matrix = pd.read_csv(gene_expression_matrices[0], delim_whitespace=True, index_col=0)
            gene_names = list(gene_expression_matrix.index)
            
            if true_times:
                t = np.array(list(gene_expression_matrix.columns)).astype('float')
            else:
                t = np.array(range(gene_expression_matrix.shape[1])).astype('float') # equally spaced time points
                        
            # scale each gene's expression across time-series to mean 0, std. dev 1.
            if not args.do_not_mean_center:
                gene_expression_matrix = pd.DataFrame(scale(np.array(gene_expression_matrix), axis=1), columns=t)
                gene_expression_matrix.index = gene_names
                
            # assign hyperpriors to noise variance:
            sigma_n2_shape = args.sigma_n2_shape # these value seems to work well in practice,
            sigma_n2_rate = args.sigma_n2_rate  # otherwise sigma_n will be (over-)optimized to zero.
            sigma_n = np.sqrt(1 / ((sigma_n2_shape + 1) * sigma_n2_rate))
            
        return(gene_expression_matrix, gene_names, sigma_n, sigma_n2_shape, sigma_n2_rate, t)

    gene_expression_matrix, gene_names, sigma_n, sigma_n2_shape, sigma_n2_rate, t = read_gene_expression_matrices_return_array_and_sigma_n(gene_expression_matrices)
    gene_expression_matrix = np.array(gene_expression_matrix)
    X = np.vstack(t)
    
    # scale t such that the mean time interval between sampling points is one unit
    # this allows reasonable initial parameters for length-scale and signal variance
    t /= np.mean(np.diff(t))

#############################################################################################

optimal_clusters_GP = {}
for cluster, genes in optimal_cluster_labels.iteritems():
    optimal_clusters_GP[cluster] = core.dp_cluster(members=genes, sigma_n=sigma_n, X=X, Y=np.array(np.mat(gene_expression_matrix[genes,:])).T, iter_num_at_birth=0)
    optimal_clusters_GP[cluster] = optimal_clusters_GP[cluster].update_cluster_attributes(gene_expression_matrix, sigma_n2_shape, sigma_n2_rate, length_scale_mu, length_scale_sigma, sigma_f_mu, sigma_f_sigma, 0, 10000, args.optimizer)

####
# Report
####

cluster_tools.save_cluster_membership_information(optimal_cluster_labels_original_gene_names, output)

####
# Plot
####

if args.plot:
    from DP_GP import plot
    if len(sim_mat) > 0:
        sim_mat_key = plot.plot_similarity_matrix(np.array(sim_mat), output_path_prefix, plot_types)
    
#     plot.plot_cluster_sizes_over_iterations(np.array(clusterings), burnIn_phaseI, burnIn_phaseII, m, output_path_prefix, plot_types)
    plot.plot_cluster_gene_expression(optimal_clusters_GP, pd.DataFrame(gene_expression_matrix, index=gene_names, columns=t), t, time_unit, output_path_prefix, plot_types)
