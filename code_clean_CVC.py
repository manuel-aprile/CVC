from gurobipy import *

import time

from networkx import *

#
#   EF_CVC_arb(G): Construct a mixed-integer extended formulation for the CVC problem on graph G and solves it
#   (set optional argument solve = 0 to only get the Gurobi model.) If solve = 1 (default), it returns [opt, t, n] with
# opt = value of the optimal solution; t = time needed to build and solve the formulation ; n = number of branch and bound nodes used by the solver.
#
#   BB_CVC(G): solves the CVC problem on G by branch and bound. Returns [S, t, n] with
#    opt = value of the optimal solution; t = total running time; n = number of nodes visited.
#
#   For more informations see ''Exact approaches for the Connected Vertex Cover problem'', Manuel Aprile



def EF_CVC_arb(G, solve =1, clique_cov=1): # Construct a mixed-integer extended formulation for the CVC problem
    s_time= time.time()
    m = Model()
    x = m.addVars(G.nodes(), vtype='b', name='x')

    r_1 = max(G.nodes(), key=lambda v: G.degree(v))
    r_2 = min(G.neighbors(r_1), key=lambda v: G.degree(v))
    n = G.number_of_nodes()

    A = []
    for (u, v) in G.edges():
        if v!= r_1 and v!= r_2:
            A.append(tuple([u, v]))
        if u!= r_1 and u!= r_2:
            A.append(tuple([v, u]))
    A.append(tuple([r_1, r_2]))

    z = m.addVars(A, vtype='b', name='z')

    l = m.addVars(G.nodes(), lb = -1, ub = n - 1, name='l')

    for (i, j) in G.edges():
        m.addConstr(x[i]+x[j]>= 1)

    for (i,j) in A:
        m.addConstr(z[(i,j)] <= x[i])
        m.addConstr(z[(i,j)] <= x[j])

    m.addConstr(sum([z[e] for e in A]) == sum([x[v] for v in G.nodes()]) - 1)

    m.addConstr(l[r_1] == 0)

    for (u, v) in A:
        m.addConstr(l[v] >= n * (z[u, v] - 1) + l[u] + x[v])

    for v in G.nodes():
        if v != r_1 and v != r_2:
            m.addConstr(sum([z[u,v] for u in G.neighbors(v)])==x[v])

    m.setObjective(sum([x[v] for v in G.nodes()]), GRB.MINIMIZE)

    if clique_cov: # Adds some clique inequalities
        cover = clique_cover(G)
        for C in cover:
            m.addConstr(sum([x[v] for v in C]) >= len(C)-1)

    if solve:
        m.optimize()
        #m.write('test_arb.lp')

        return [m.ObjVal, time.time()-s_time, m.NodeCount]

    else:
        return m


def clique_cover(G): # Greedily constructs a sets of maximal cliques covering all the edges of the graph

    G_cov=G.copy()

    my_clique_cover = []
    while G_cov.edges():
        v = max(G.nodes(), key=lambda u : G_cov.degree(u))
        C = max(list(find_cliques(G.subgraph(list(G_cov.neighbors(v))+[v]))), key=len)
        my_clique_cover.append(C)

        for e in G.subgraph(C).edges():
            if e in G_cov.edges():
                G_cov.remove_edge(*e)
    return my_clique_cover


def BB_CVC(G, time_limit=3600, heur_start=True): # Branch and bound algorithm for the CVC problem
    num_nodes = 0
    time_s = time.time()
    S_best = []
    C = articulation_points(G)

    if heur_start:
        S_best = heuristic_solution(G)

    vtx = sort_degree(G.subgraph(v for v in G.nodes() if v not in C).copy(), reverse =False)

    A = [[[],vtx ]]
    while A:
        [S, U] = A.pop()

        while U and len(S_best)< len(S) + greedy_col_comp(G,U):
            num_nodes += 1
            v = U.pop()
            A.append([S, U])
            S = S+[v]
            C = list(articulation_points(G.subgraph([u for u in G.nodes() if u not in S])))
            U = [u for u in U if not G.has_edge(u,v) and u not in C]

            if len(S_best)<len(S):
                    S_best = S
            if time.time() - time_s>= time_limit:
                return [S_best, time.time() - time_s,num_nodes, False]
    return [G.number_of_nodes() - len(S_best), round(time.time() - time_s,1), num_nodes,True]

def greedy_col_comp(G,K): # bound on the stability number of the subgraph of G induced by K, obtained via greedy coloring
    return max(greedy_color(complement(G.subgraph(K))).values())+1


def heuristic_solution(G): #heuristically find a connected vertex cover
    S = algorithms.approximation.clique.maximum_independent_set(G)
    comp = list(connected_components(G.subgraph([v for v in G.nodes() if v not in S])))
    while len(comp) > 1:
        v = max(S,key = lambda v : len([c for c in comp if [u for u in G.neighbors(v) if u in c]]))
        S.remove(v)
        comp = list(connected_components(G.subgraph([v for v in G.nodes() if v not in S])))
    return list(S)


def sort_degree(G, reverse = False): # order the nodes of the graph by iteratively finding the node of minimum (or maximum) degree
    nodes = []
    while G.number_of_nodes()>0:
        if reverse:
            v = max(G.nodes(), key=G.degree())
        else:
            v = min(G.nodes(), key=G.degree())
        G.remove_node(v)
        nodes.append(v)
    return nodes

