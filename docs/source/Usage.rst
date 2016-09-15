Usage
=====
Create gene-by-gene posterior similarity matrix according to expression over time course and find optimal clustering of genes by expression over time course (and helpful plots)::
    
    DP_GP_cluster.py -i expression.txt -o output_prefix [ optional args, e.g. --plot ]
    
Different clustering criteria may be applied after Gibbs sampling to yield different sets of clusters according to an alternative optimality criterion.
Also, if `--plot` flag not indicated when the `DP_GP_cluster.py` script is called, plots can be generated post-sampling::

    DP_GP_cluster_post_gibbs_sampling.py -i expression.txt \
    --sim_mat output_prefix_posterior_similarity_matrix.txt \
    --clusterings output_prefix_clusterings.txt \
    --criterion MPEAR \
    --plot --plot_types 
    --output output_prefix_MPEAR_optimal_clustering.txt \
    --output_path_prefix output_prefix_MPEAR