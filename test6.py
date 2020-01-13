#mport pytest
import math
import os
import networkx as nx
import itertools
import  graph_utils
import time
import numpy as nm
import read_pajek as paj
import eigen_graph as eg
import community
import parallelism
import trajanovski as traj
import matplotlib.pyplot as plt
from graph_utils import simm_matrix_2_graph
from eigen_graph import eigen_centrality, vec_dist, build_matrix, \
    sample_simm_matrix, sample_simm_matrix2, modularity_clone_matrix, \
    laplacian_clone_matrix, generate_matrix, synthetic_modularity_matrix, \
    leading_eigenvector,modularity_clone_graph

from entropy_analysis import entropy_analysis
import graph_utils as gu
import kl_modularity
import entropy_analysis



import dc_sbm as sbm


count=0
m_bg_result = []
c_a_bg_result=[]
partiton_ratio=[]
d_pcc=[]
for i in range(10):
    count+=1
    if count<=10:
        #G = nx.read_weighted_edgelist("D:/software/python/spectral_graph_forge-master/spectral_graph_forge-master/toy.edges")
        #G = nx.read_weighted_edgelist("C:/Users/Administrator/Desktop/lee/ego-Facebook/facebook/3980.edges")
        #G = nx.read_weighted_edgelist("C:/Users/Administrator/Desktop/lee/ego-Facebook/facebook/348.edges")
        G = nx.karate_club_graph()
        #G=nx.generators.davis_southern_women_graph()
        #G = nx.generators.florentine_families_graph()
        A = nx.to_numpy_matrix(G)
        B_matrix = modularity_clone_matrix(A, 6)
       #B_matrix =entropy_analysis.truncated_mod_approx(A, 6)
       #B_1=entropy_analysis.sample_mod_matrix(B_matrix)
        B = simm_matrix_2_graph(B_matrix, None)
        print ("***************%dth running result***************" %(i+1))
        print "***************now is the modularity of the original Graph B"
        m_b = kl_modularity.maximum_modularity(B)[0]
        print "***************now is the modularity of the original Graph G"
        m_g = kl_modularity.maximum_modularity(G)[0]
        print "***************this is the modularity ratio of the original Graph G and new graph B"
        print float(m_b / m_g)
        m_bg_result.append(float(m_b / m_g))

        print "##########now is the clustering########## "
        c_g = graph_utils.clustering(A)
        c_b = graph_utils.clustering(B_matrix)
        print "this is the clustering  of the original Graph G and new graph B"
        print c_b
        print c_g
        print "##########now is the degree sequence########## "
        d_g = graph_utils.get_degrees(A)
        d_b = graph_utils.get_degrees(B_matrix)
        print "this is the degree  of the original Graph G and new graph B"
        print d_b
        print d_g
        cov=nm.cov(d_b,d_g,ddof=0)[0][1]
        standard_d_b=nm.std(d_b,ddof=0)
        standard_d_g = nm.std(d_g, ddof=0)
        ppcc=cov/(standard_d_b*standard_d_g)
        print(str(ppcc))
        d_pcc.append(ppcc)
        print(str(cov))
        print "##########now is the partition##########"
        p_g = community.best_partition(G)
        p_b = community.best_partition(B)
        c_a_g = graph_utils.average_clustering(A)
        c_a_b = graph_utils.average_clustering(B_matrix)
        print "this is the partition  of the original Graph G and new graph B"
        print p_b
        print p_g
        list1 = list(p_b.values())
        list2 = list(p_g.values())
        temp= 0
        for i in range(len(list1)):
            if list1[i] == list2[i]:
                temp += 1
        print float(temp) / len(list1)
        partiton_ratio.append(float(temp) / len(list1))
        print "this is the avg clustering of the original Graph G and new graph B"
        print c_a_b
        print c_a_g
        c_a_bg_result.append(float(c_a_b/ c_a_g))
    else:
        break
print m_bg_result
sum=0
for i in range(len(m_bg_result)):
    sum+=m_bg_result[i]
print sum/(len(m_bg_result))


print partiton_ratio
sum=0
for i in range(len(partiton_ratio)):
    sum+=partiton_ratio[i]
print sum/(len(partiton_ratio))



print c_a_bg_result
sum=0
for i in range(len(c_a_bg_result)):
    sum+=c_a_bg_result[i]
print sum/(len(c_a_bg_result))


print d_pcc
sum=0
for i in range(len(d_pcc)):
    sum+=d_pcc[i]
print sum/(len(d_pcc))