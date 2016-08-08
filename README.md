
## DP_GP_cluster

DP_GP_cluster clusters genes by expression over a time course using a Dirichlet process Gaussian process model.

## Code Example

Create gene-by-gene posterior similarity matrix according to expression over time course and find optimal clustering of genes by expression over time course (and helpful plots):
    
    DP_GP_cluster.py -i expression.txt -o output_prefix [ optional args, e.g. -n 1000 --true_times --criterion MAP... --plot ]
    
Different clustering criteria may be applied after Gibbs sampling to yield different sets of clusters. Also, if `--plot` flag not indicated when the above script is called, plots can be generated post-sampling:

    DP_GP_cluster_post_gibbs_sampling.py -i expression.txt \
    --sim_mat output_prefix_posterior_similarity_matrix.txt \
    --clusterings output_prefix_clusterings.txt \
    --criterion MPEAR \
    --plot --plot_types png,pdf \
    --output output_prefix_MPEAR_optimal_clustering.txt \
    --output_path_prefix output_prefix_MPEAR
    
## Motivation

Genes that follow similar expression trajectories in response to stress or stimulus tend to share biological functions.  Thus, it is reasonable and common to cluster genes by expression trajectories.  Two important considerations in this problem are (1) selecting the "correct" or "optimal" number of clusters and (2) modeling the trajectory and time-dependency of gene expression. A [Dirichlet process](http://en.wikipedia.org/wiki/Dirichlet_process) can determine the number of clusters in a nonparametric manner, while a [Gaussian process](http://en.wikipedia.org/wiki/Gaussian_process) can model the trajectory and time-dependency of gene expression in a nonparametric manner.

## Installation and Dependencies

DP_GP_cluster requires the following Python packages:
    
    GPy, pandas, numpy, scipy (>= 0.14), matplotlib.pyplot

It has been tested in Python 2.7 on CentOS v.6.7 and with Anaconda distributions of the latter four above packages.

Download source code and uncompress, then:

    python setup.py install

## Tests

Describe and show how to run the tests with code examples.

## Lab notebooks

to be updated, ignore the below...

[Methods](http://nbviewer.ipython.org/gist/IanMcDowell/24429d1816f7002c2558)
Results comparison
Results A549 dexamethasone
Results halobacteria

## Citation

    bibtex citation of publication

## License

A short snippet describing the license