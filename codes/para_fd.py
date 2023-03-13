# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:36:14 2022

@author: khati
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import copy
import heapq
from collections import namedtuple
import alite_fd as cfd
import time


class UnionFind:
    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root
        
    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest

Partition = namedtuple('Partition', ['included', 'excluded'])


def getGraphCost(graph):
    '''Returns the total sum of weights of edges in a graph'''
    return sum(e[0] for e in graph)

def kruskal(edges):
    '''Returns the mst'''
    edges.sort()
    
    subtrees = UnionFind()
    solution = []

    for edge in edges:
        _, u, v = edge
        if subtrees[u] != subtrees[v]:
            solution.append(edge)
            subtrees.union(u, v)

    return solution

class IncreasingMST():
    def __init__(self, edges):
        self.edges = edges
        self.edges.sort()  # Sort edges by weight
        self.order = self.__getNbVertices(edges)
    
    @classmethod
    def __getNbVertices(self, edges):
        vertices = set()
        for c, u, v in edges:
            vertices.add(u)
            vertices.add(v)
        return len(vertices)
    
    @classmethod
    def __partition(self , mstEdges , p):
        '''
        Given an MST and a __partition P, P partitions the vertices of the MST.
        The union of the set of partitions generated here with the last MST 
        is equivalent to the last __partition as an input.
        '''
 
        p1 = copy.deepcopy(p)
        p2 = copy.deepcopy(p)
 
        partitions = []
        
        # Find open edges : they are in MST but are not included nor excluded 
        open_edges = [edge for edge in mstEdges if edge not in p.included 
                      and edge not in p.excluded]    
 
        for e  in open_edges :
            p1.excluded.append(e)
            p2.included.append(e)
            
            partitions.append(p1)
            p1 = copy.deepcopy(p2)
 
        return partitions
    
    def __kruskalMST(self, P):
        '''
        Returns the MST of a graph contained in the __partition P.
        Returns None if there is not any.
        P [0] -> edges included, P [1] -> edges excluded
        '''

        # Initialize the subtrees for Kruskal Algorithm with included edges
        subtrees = UnionFind()
        for c, u , v in P.included :
            subtrees.union(u , v)
        
        # Find open Edges
        edges = [e for e in self.edges if e not in P.included and e not in P.excluded]
        
        # Create a solution tree with the list of included edges from the __partition
        tree = list(P.included)
        
        # Add edges connecting vertices not connected yet, until we get to
        # A solution, or discover that you can not connect all vertices
        
        # Apply Kruskal on open edges
        for edge in edges :
            c, u , v = edge
            if subtrees [u] != subtrees [v]:
                tree.append(edge)
                subtrees.union(u, v)
        
        if self.order == self.__getNbVertices(tree) and \
        len(tree) == self.order - 1:
            return tree
        else:
            return None
    
    def mst_iter(self):
        '''Minimum spanning tree iterator : yields the MST in order of their cost
        Warning this quickly becomes computer intensive both on CPU and memory'''
        mst = kruskal(self.edges)

        mstCost = getGraphCost(mst)
        List = [(mstCost, Partition([], []), mst)]
        
        # While we have a __partition
        while len(List):
            # Get __partition with the smallest spanning tree
            # and remove it from the list
            cost, partition, tree = heapq.heappop(List)
            
            # Yield MST result
            yield tree
               
            edges = tree
            
            # Partition previous partition 
            newPartitions = self.__partition(edges, partition)
            for p in newPartitions:
                tree = self.__kruskalMST(p)
                if tree:
                    # print 'isTree', tree
                    cost = getGraphCost(tree)
                    heapq.heappush(List, (cost, p, tree))
        


print("Enter input folder path:")    
#input_path = str(input())
input_path = r"minimum_example/"
print(input_path)
#input_path = r"input/"
output_path = r"output_tables/parallelfd_"+ input_path
# =============================================================================
# if not os.path.exists(output_path):
#   # Create a new directory because it does not exist 
#   os.makedirs(output_path)
#   print("output directory is is created!")
# =============================================================================
stat_path = r"statistics/parallelfd_"+ input_path[:-1]+".csv"
foldernames = glob.glob(input_path + "*")
statistics = pd.DataFrame(
            columns = ["cluster", "n", "s", "f", "total_cols", "labeled_nulls",
                       "produced_nulls", "spanning_trees",
                       "subsume_time",
                       "subsumed_tuples", "total_time", "f_s_ratio"])

#G = [(1, 'A', 'B'), (1, 'B', 'C'), (1, 'C', 'A')]
for cluster in foldernames:
        try:
            filenames = glob.glob(cluster + "/*.csv")
        except:
            continue
        cluster_name = cluster.rsplit(os.sep)[-1]
        m = len(filenames)
        all_columns_order = set()
# =============================================================================
#             create scheme graphs
# =============================================================================
        start_time = time.time_ns()        
        order = []
        for file in filenames:    
            df= pd.read_csv(file, nrows=0)
            df.columns = map(str.lower, df.columns)
            order.append(set(df.columns))
            all_columns_order = all_columns_order.union(set(df.columns))
        total_tables = len(filenames)
        all_columns_order = list(all_columns_order)
        spanning_graph = []
        #np.fill_diagonal(scheme_graph, 1)
        for i in range(0, total_tables-1):
            for j in range(i, total_tables):
                if i != j and len(order[i].intersection(order[j]))>0:
                    spanning_graph.append((1, str(i), str(j)))
        
        iMST = IncreasingMST(spanning_graph)
        #print ('\nMinimum Spanning trees in order of increasing cost:')
        all_integration_orders = []
        for tree in iMST.mst_iter():
            first_edge = tree[0]
            outer_join_order = (first_edge[1], first_edge[2])
            for branch in tree[1:]:
                if branch[1] not in outer_join_order:
                    outer_join_order = outer_join_order + (branch[1],) 
                if branch[2] not in outer_join_order:
                    outer_join_order = outer_join_order + (branch[2],)
            
            #print (tree)
            #print(outer_join_order)
            if outer_join_order not in all_integration_orders:
                all_integration_orders.append(outer_join_order)
        all_produced_tuples = set()
        original_filenames = filenames
        for each in all_integration_orders:
            new_order = []
            for node in each:
                new_order.append(original_filenames[int(node)])
            #print(each)
            #print(new_order)
            #filenames = new_order
            filenames = new_order
            null_set = set()
            null_count = 0
            table1 = pd.read_csv(filenames[0], encoding='latin1',warn_bad_lines=True, error_bad_lines=False)
            table1 = table1.drop_duplicates().reset_index(drop=True)
            table1 = table1.replace(r'^\s*$',np.nan, regex=True)
            table1 = table1.replace("-",np.nan)
            table1 = table1.replace(r"\N",np.nan)
            if table1.isnull().sum().sum() > 0:
                #print(filenames[0])
                table1, null_count, current_null_set = cfd.ReplaceNulls(table1, null_count)
                null_set = null_set.union(current_null_set)
            table1 = cfd.preprocess(table1)
            total_join_itr = 1
            s = table1.shape[0]
            labeled_nulls = len(null_set)
            for file in filenames[1:]:
                table2 = pd.read_csv(file, encoding='latin1',warn_bad_lines=True, error_bad_lines=False)
                s += table2.shape[0]
                table2 = table2.drop_duplicates().reset_index(drop=True)
                table2 = table2.replace(r'^\s*$',np.nan, regex=True)
                table2 = table2.replace("-",np.nan)
                table2 = table2.replace(r"\N",np.nan)
                if table2.isnull().sum().sum() > 0:
                    #print(filenames[0])
                    table2, null_count, current_null_set = cfd.ReplaceNulls(table2, null_count)
                    null_set = null_set.union(current_null_set)
                table2 = cfd.preprocess(table2)
                #print(table2)
                table1 = table1.merge(table2, how = "outer", on = None)
                total_join_itr += 1
                #print("outer joined up to table :", total_join_itr)
                if table1.isnull().sum().sum() > 0:
                    #print(filenames[0])
                    table1, null_count, current_null_set = cfd.ReplaceNulls(table1, null_count)
                    null_set = null_set.union(current_null_set)
                table1 = cfd.preprocess(table1)
            #print(table1)
            if len(null_set) > 0:
                table1 =  cfd.AddNullsBack(table1, null_set)
            #print(table1)
            lower_case_column_order = [x.lower() for x in all_columns_order]
            tablex = table1.reindex(columns=lower_case_column_order)
            #print(table1)
            outer_tuples = {tuple(x) for x in tablex.values}
            #print(outer_tuples)
            for x in outer_tuples:
                all_produced_tuples.add(x)
        start_subsume_time = time.time_ns()
        subsumed_tuples_list = cfd.EfficientSubsumption(list(all_produced_tuples))
        end_subsume_time = time.time_ns()
        subsume_time = int(end_subsume_time - start_subsume_time)/ 10**9
        total_time = int(end_subsume_time - start_time)/10**9
        subsumed_tuples = len(list(all_produced_tuples)) - len(subsumed_tuples_list)
        #table1_data = [tuple(x) for x in table1.values]
        f = len(subsumed_tuples_list)
        produced_nulls = cfd.CountProducedNulls(subsumed_tuples_list)
        spanning_trees = len(all_integration_orders)
        total_cols = len(lower_case_column_order)
        #outer_tuples = {tuple(x) for x in table1.values}
        print("Finished upto cluster", cluster)
        print("Total Tuples:", f)
        print("Total nulls:", produced_nulls)
        print("-----------------------------")
        print("Output tuples:")
        for t in subsumed_tuples_list:
            print(t)
        #result_FD = pd.DataFrame(subsumed_tuples_list, columns =lower_case_column_order)
        #result_FD = result_FD.replace(np.nan, "nan", regex = True)
        #print(result_FD)
        #result_FD.to_csv(output_path+ cluster_name+".csv",index = False)
        append_list = [cluster_name, m, s, f, total_cols, labeled_nulls,
                   produced_nulls, spanning_trees,
                   subsume_time, subsumed_tuples, total_time, f/s]
        a_series = pd.Series(append_list, index = statistics.columns)
        statistics = statistics.append(a_series, ignore_index=True)    
        #statistics.to_csv(stat_path, index = False)
