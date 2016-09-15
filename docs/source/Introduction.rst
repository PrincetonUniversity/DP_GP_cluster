Introduction
==================

Genes that follow similar expression trajectories in response to stress or stimulus tend to share biological functions.  Thus, it is reasonable and common to cluster genes by expression trajectories.  Two important considerations in this problem are (1) selecting the "correct" or "optimal" number of clusters and (2) modeling the trajectory and time-dependency of gene expression. A `Dirichlet process <http://en.wikipedia.org/wiki/Dirichlet_process>`_ can determine the number of clusters in a nonparametric manner, while a `Gaussian process <http://en.wikipedia.org/wiki/Gaussian_process>`_ can model the trajectory and time-dependency of gene expression in a nonparametric manner.

DP_GP_cluster clusters genes by expression over a time course using an infinite Gaussian process mixture model.