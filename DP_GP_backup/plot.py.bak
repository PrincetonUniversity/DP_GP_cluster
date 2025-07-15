'''
Created on 2015-03-27

@author: Ian McDowell, Dinesh Manandhar
'''
# Force matplotlib to not use any Xwindows backend, which should enable
# the script to be run on high-performance computing cluster without error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as sch
import numpy as np
import collections
import GPy

def adjust_spines(ax, spines):
    ''' 
    see matplotlib examples:
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html
    ''' 
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
        else:
            spine.set_color('none')  # don't draw spine
    
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def plot_cluster_gene_expression(clusters, gene_expression_matrix, t, t_labels, time_unit, output_path_prefix, plot_types, unscaled, do_not_mean_center):
    ''' 
    Plot gene expression over a time course with a panel for each cluster. Each panel contains
    transparent red lines for the expression of each individual gene within the cluster, the
    cluster mean, and a ribbon twice the standard deviation about the cluster mean.  This is
    essentially a wrapper function for GPy.plotting.matplot_dep.base_plots.gpplot.
    
    :param clusters: dictionary of dp_cluster objects
    :type clusters: dict
    :param gene_expression_matrix: expression over timecourse of dimension |genes|x|timepoints|
    :type gene_expression_matrix: pandas dataframe
    :param t: sampled timepoints
    :type t: numpy array of floats
    :param time_unit: time units in reference to t, e.g. 'days','hr.','min.','sec.'
    :type t: str
    :param output_path_prefix: absolute path to output
    :type output_path_prefix: str
    :param plot_types: plots to be generated, e.g. ['png','pdf','svg'] or simply ['png']
    :type plot_types: list of strings
    
    :rtype: None (output is saved to file(s))
    
    ''' 
    # cluster IDs:
    IDs = sorted(clusters)
    # one panel per cluster:
    total_subplots = len(IDs)
    # max of 6 panels per figure or page
    subplots_per_fig = 6
    total_no_of_figs = int(np.ceil(total_subplots/float(subplots_per_fig)))
    total_cols = 2 # generate this many columns of subplots in the figure.
    total_rows = np.ceil(subplots_per_fig/total_cols) # each figure generate will have this many rows.
    IDs_split = [IDs[i:i+subplots_per_fig] for i in xrange(0, len(IDs), subplots_per_fig)]
    index = 1
    for c, IDs in enumerate(IDs_split):
        fig = plt.figure(num=None, figsize=(8,12), dpi=300, facecolor='w', edgecolor='k') #figsize=(12,8),
        for i, ID in enumerate(IDs):
            ax = fig.add_subplot(total_rows, total_cols, i+1)
            # create a range of values at which to evaluate the covariance function
            Xgrid = np.vstack(np.linspace(min(t), max(t), num=500))
            # calculate mean and variance at grid of x values
            mu, v = clusters[ID].model.predict(Xgrid, full_cov=False, kern=clusters[ID].model.kern)
            mu = np.hstack(mu.mean(axis=1))
            v = v[:,0]
            GPy.plotting.matplot_dep.base_plots.gpplot(Xgrid, mu, mu - 2*v**(0.5),  mu + 2*v**(0.5), ax=ax)
            ax.set_xlim((min(t),max(t)))
            if ( not unscaled ) and ( not do_not_mean_center ) :
                ax.set_ylim((-3,3))
            
            # plot an x-axis at zero
            plt.axhline(0, color='black', ls='--', alpha=0.5)
            # plot the expression of each gene in the cluster
            for gene in list(clusters[ID].members):
                ax.plot(t, np.array(gene_expression_matrix.ix[gene]), color='red', alpha=0.1)
            
            # plot mean expression of cluster
            ax.plot(Xgrid, mu, color='blue')
            # create legend
            light_blue_patch = mpatches.Rectangle([0, 0], 1, 1, facecolor='#33CCFF', edgecolor='blue', lw=1, alpha=0.3)
            red_line = mlines.Line2D([], [], color='red', label='individual gene trajectory')
            ax.legend([ax.lines[0], light_blue_patch, red_line], \
                      ['cluster mean', u'cluster mean \u00B1 2 x std. dev.', 'individual gene trajectory'], 
                      loc=4, frameon=False, prop={'size':6})
            # prettify axes
            adjust_spines(ax, ['left', 'bottom'])
            # label x-axis
            if time_unit == '':
                ax.set_xlabel("Time")
            else:
                ax.set_xlabel("Time in %s"%(time_unit))
            ax.set_xticks(t)
            ax.set_xticklabels(t_labels)
            ax.set_ylabel('Gene expression')
            ax.set_title('Cluster %s'%(index))
            index+=1
        
        plt.tight_layout()
        
        for plot_type in plot_types:
            plt.savefig(output_path_prefix + '_gene_expression_fig_' + str(c+1) + '.' + plot_type)

#############################################################################################

def plot_similarity_matrix(sim_mat, output_path_prefix, plot_types):
    ''' 
    Plot the posterior similarity matrix as heatmap with dendrogram. 
    
    dim(S) = n x n, where n = total number of genes.
    S[i,j] = (# samples gene i in cluster with gene j)/(# total samples)
    
    Hierarchically cluster by complete linkage for orderliness of visualization.
    Function returns all gene names in the order in which they were clustered/displayed.
    This list might be used to visually inspect heatmap,
    yet heatmap is largely intended for high-level view of data.
    
    :param sim_mat: sim_mat[i,j] = (# samples gene i in cluster with gene j)/(# total samples)
    :type sim_mat: numpy array of (0-1) floats
    :param output_path_prefix: absolute path to output
    :type output_path_prefix: str
    :param plot_types: plots to be generated, e.g. ['png','pdf','svg'] or simply ['png']
    :type plot_types: list of strings
    
    :rtype: array-like, names of genes in order of clustering (both left to right and top to bottom)
    
    '''
    dist_mat = 1 - sim_mat
    
    sch.set_link_color_palette(['black'])
    
    # Compute and plot left dendrogram.
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_axes([0,0.02,0.2,0.6])
    Y = sch.linkage(dist_mat, method='complete')
    # color_threshold=np.inf makes dendrogram black
    Z = sch.dendrogram(Y, orientation='left', link_color_func=lambda x: 'black' ) 
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    
    ax1.invert_yaxis()
    #Compute and plot the heatmap
    axmatrix = fig.add_axes([0.2,0.02,0.6,0.6])
    
    # reorder similarity matrix by linkage
    idx = Z['leaves']
    sim_mat = sim_mat[idx,:]
    sim_mat = sim_mat[:,idx]
    
    im = axmatrix.matshow(sim_mat, aspect='auto', origin='lower', cmap="cubehelix", vmax=1, vmin=0)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    axmatrix.invert_yaxis()
    
    # Plot colorbar.
    axcolor = fig.add_axes([0.81,0.02,0.02,0.6])
    cbar = plt.colorbar(im, cax=axcolor)
    cbar.ax.set_ylabel('Proportion of Gibbs samples in which row i and column j were co-clustered', rotation=270, labelpad=10)
    fig.subplots_adjust(wspace=0, hspace=0)
    for plot_type in plot_types:
        plt.savefig(output_path_prefix + "_posterior_similarity_matrix_heatmap." + plot_type, bbox_inches=0)
    
    return(idx)

#############################################################################################

def plot_cluster_sizes_over_iterations(all_clusterings, burnIn_phaseI, burnIn_phaseII, m, output_path_prefix, plot_types):
    ''' 
    Plot size of clusters over GS iterations after burn-in phase I 
    where x-axis is iteration and vertical thickness of band is proportional to 
    size of cluster. Each cluster has unique color. Burn-in phase II
    is indicated with a vertical line.
    
    :param all_clusterings: all_clusterings[i,j] is the cluster to which gene j belongs at iteration i
    :type all_clusterings: numpy array of ints
    :param burnIn_phaseI: iteration at which first burn-in phase ends
    :type burnIn_phaseI: int
    :param burnIn_phaseII: iteration at which second burn-in phase ends
    :type burnIn_phaseII: int
    :param m: number of "empty table" at each Gibbs sampling iteration
    :type m: int
    :param output_path_prefix: absolute path to output
    :type output_path_prefix: str
    :param plot_types: plots to be generated, e.g. ['png','pdf','svg'] or simply ['png']
    :type plot_types: list of strings
    
    :rtype: None (output is saved to file(s))
    
    '''
    # strip off "header" of array, which is a vector of gene names
    gene_names = all_clusterings[0,:]
    # now, array only contains the cluster number for each iteration (row) for each gene (col)
    all_clusterings = all_clusterings[1:,:].astype('int')
    
    highest_cluster_number = np.max(all_clusterings) - m
    all_clusterIDs = list(set(np.unique(all_clusterings)) - set(range(m)))
    
    # for each cluster and across all iterations, find the maximum size that cluster attained
    max_num_genes_per_cluster = {}
    for clusterID in all_clusterIDs:
        GSiters, genes = np.where(all_clusterings == clusterID)
        counts = collections.Counter(GSiters)
        [(GSiter, number)] = counts.most_common(1)
        max_num_genes_per_cluster[clusterID] = number
    
    # find the size of each cluster over iterations
    cluster_size_over_iterations = {}
    for iter_num in range(len(all_clusterings)):
        cluster_size_over_iterations[iter_num] = collections.Counter(all_clusterings[iter_num,:])
    
    total_height = sum(max_num_genes_per_cluster.values())
    # height will decrement through the plotting of clusters
    current_height = total_height
    # set-up the figure:
    fig = plt.figure(figsize=(8,8), dpi=300, facecolor='w', edgecolor='k')
    a = fig.add_subplot(1,1,1)
    # assign each cluster a unique color
    color=iter(plt.cm.rainbow(np.linspace(0,1,len(all_clusterIDs))))
    for cluster in all_clusterIDs:
        c=next(color)
        mid_line_y = current_height - max_num_genes_per_cluster[cluster] * 0.5
        half_num_genes_in_clust_over_iters = np.zeros(len(cluster_size_over_iterations.keys()))
        for iter_num in sorted(cluster_size_over_iterations):
            if cluster in cluster_size_over_iterations[iter_num]:
                half_num_genes_in_clust_over_iters[iter_num] = cluster_size_over_iterations[iter_num][cluster] * 0.5
        a.fill_between(np.array(sorted(cluster_size_over_iterations)) + burnIn_phaseI, \
                         mid_line_y-half_num_genes_in_clust_over_iters, \
                         mid_line_y+half_num_genes_in_clust_over_iters, \
                         facecolor=c, alpha=0.75)
        current_height = current_height - max_num_genes_per_cluster[cluster]
    
    a.set_xlabel("Iterations")
    a.set_ylabel("Cluster size")
    plt.axvline(burnIn_phaseII, alpha = 0.5, color = 'b', label = "Burn-in phase II ends")
    plt.tight_layout()
    leg = plt.legend(prop={'size':6})
    leg.get_frame().set_linewidth(0.0)
    for plot_type in plot_types:
        plt.savefig(output_path_prefix + "_cluster_sizes." + plot_type, bbox_inches=0)
