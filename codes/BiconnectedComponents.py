# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:07:45 2022

@author: khati
"""

# Python program to find biconnected components in a given
# undirected graph
# Complexity : O(V + E)


from collections import defaultdict
import sys
# This class represents an directed graph
# using adjacency list representation
class Graph:

	def __init__(self, vertices):
		# No. of vertices
		self.V = vertices
		
		# default dictionary to store graph
		self.graph = defaultdict(list)
		
		# time is used to find discovery times
		self.Time = 0
		
		# Count is number of biconnected components
		self.count = 0
        
		self.biconnected_components = []
		self.articulation_points = []
	# function to add an edge to graph
	def addEdge(self, u, v):
		self.graph[u].append(v)
		self.graph[v].append(u)

	'''A recursive function that finds and prints strongly connected
	components using DFS traversal
	u --> The vertex to be visited next
	disc[] --> Stores discovery times of visited vertices
	low[] -- >> earliest visited vertex (the vertex with minimum
			discovery time) that can be reached from subtree
			rooted with current vertex
	st -- >> To store visited edges'''
	def BCCUtil(self, u, parent, low, disc, st):

		# Count of children in current node
		children = 0

		# Initialize discovery time and low value
		disc[u] = self.Time
		low[u] = self.Time
		self.Time += 1


		# Recur for all the vertices adjacent to this vertex
		for v in self.graph[u]:
			# If v is not visited yet, then make it a child of u
			# in DFS tree and recur for it
			if disc[v] == -1 :
				parent[v] = u
				children += 1
				st.append((u, v)) # store the edge in stack
				self.BCCUtil(v, parent, low, disc, st)

				# Check if the subtree rooted with v has a connection to
				# one of the ancestors of u
				# Case 1 -- per Strongly Connected Components Article
				low[u] = min(low[u], low[v])

				# If u is an articulation point, pop
				# all edges from stack till (u, v)
				if parent[u] == -1 and children > 1 or parent[u] != -1 and low[v] >= disc[u]:
					self.count += 1 # increment count
					w = -1
					current_components = []
					while w != (u, v):
						w = st.pop()
						current_components.append(w)
						#print(w,end=" ")
					#print()
					self.biconnected_components.append(current_components)
			
			elif v != parent[u] and low[u] > disc[v]:
				'''Update low value of 'u' only of 'v' is still in stack
				(i.e. it's a back edge, not cross edge).
				Case 2
				-- per Strongly Connected Components Article'''

				low[u] = min(low [u], disc[v])
	
				st.append((u, v))


	# The function to do DFS traversal.
	# It uses recursive BCCUtil()
	def BCC(self):
		
		# Initialize disc and low, and parent arrays
		disc = [-1] * (self.V)
		low = [-1] * (self.V)
		parent = [-1] * (self.V)
		st = []

		# Call the recursive helper function to
		# find articulation points
		# in DFS tree rooted with vertex 'i'
		for i in range(self.V):
			if disc[i] == -1:
				self.BCCUtil(i, parent, low, disc, st)

			# If stack is not empty, pop all edges from stack
			if st:
				self.count = self.count + 1
				current_components = []
				while st:
					w = st.pop()
					current_components.append(w)
					#print(w,end=" ")
				#print ()
				self.biconnected_components.append(current_components)
	def APUtil(self, u, visited, ap, parent, low, disc):
        # Count of children in current node
		children = 0

		# Mark the current node as visited and print it
		visited[u]= True

		# Initialize discovery time and low value
		disc[u] = self.Time
		low[u] = self.Time
		self.Time += 1

		# Recur for all the vertices adjacent to this vertex
		for v in self.graph[u]:
			# If v is not visited yet, then make it a child of u
			# in DFS tree and recur for it
			if visited[v] == False :
				parent[v] = u
				children += 1
				self.APUtil(v, visited, ap, parent, low, disc)

				# Check if the subtree rooted with v has a connection to
				# one of the ancestors of u
				low[u] = min(low[u], low[v])

				# u is an articulation point in following cases
				# (1) u is root of DFS tree and has two or more children.
				if parent[u] == -1 and children > 1:
					ap[u] = True

				#(2) If u is not root and low value of one of its child is more
				# than discovery value of u.
				if parent[u] != -1 and low[v] >= disc[u]:
					ap[u] = True
					
				# Update low value of u for parent function calls
			elif v != parent[u]:
				low[u] = min(low[u], disc[v])
    

    # The function to do DFS traversal. It uses recursive APUtil()
	def AP(self):

		# Mark all the vertices as not visited
		# and Initialize parent and visited,
		# and ap(articulation point) arrays
		visited = [False] * (self.V)
		disc = [float("Inf")] * (self.V)
		low = [float("Inf")] * (self.V)
		parent = [-1] * (self.V)
		ap = [False] * (self.V) # To store articulation points

		# Call the recursive helper function
		# to find articulation points
		# in DFS tree rooted with vertex 'i'
		for i in range(self.V):
			if visited[i] == False:
				self.APUtil(i, visited, ap, parent, low, disc)

		for index, value in enumerate (ap):
			if value == True:
        			#print(index, end = " ")
        			self.articulation_points.append(index)

def FindArticulationPointsAndBiconnectedComponents(graph_edges, total_nodes):
    g = Graph(total_nodes)
    for item in graph_edges:
        g.addEdge(item[0], item[1])
    g.AP()
    g.BCC()
    return g.articulation_points, g.biconnected_components

