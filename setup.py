from distutils.core import setup 
from Cython.Distutils import build_ext 
from distutils.extension import Extension 
import numpy

# ext_modules = [Extension("adjust/solvers_opt", ["adjust/solvers_opt.pyx"])] 

ext_modules = [Extension("DP_GP.core", ["DP_GP/core.pyx"], 
                         include_dirs=[numpy.get_include()]),
               Extension("DP_GP.cluster_tools", ["DP_GP/cluster_tools.pyx"], 
                         include_dirs=[numpy.get_include()])]
                         
setup(
      name='DP_GP_cluster',
      version='0.1',
      description='Tools for gene expression similarity matrix generation and clustering by Dirichlet Process Gaussian Process.',
      url='https://github.com/ReddyLab/DP_GP_cluster',
      author='Ian McDowell, Dinesh Manandhar, Barbara Engelhardt',
      author_email='ian.mcdowell@duke.edu',
      keywords = ['clustering','Dirichlet Process', 'Gaussian Process', \
                  'Bayesian', 'gene expression', 'similarity matrix'],
      license='Duke',
      packages=['DP_GP'],
      ext_modules = ext_modules, 
      cmdclass = {'build_ext': build_ext},
      package_dir={'DP_GP':'DP_GP'},
      scripts=['bin/DP_GP_cluster.py', 'bin/DP_GP_cluster_post_gibbs_sampling.py'],
      long_description=open('README.md', 'rt').read()
     )
