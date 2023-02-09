from gurobipy import *

import time

from networkx import *

#
#   EF_CVC_arb(G): Construct a mixed-integer extended formulation for the CVC problem on graph G and solves it
#   (set optional argument solve = 0 to only get the Gurobi model.) If solve = 1 (default), it returns [opt, t, n] with
# opt = value of the optimal solution; t = time needed to build and solve the formulation ; n = number of branch and bound nodes used by the solver.
#
#   RDS_CVC(G): solves the CVC problem on G by applying Russian Doll Search.
#
#   RDS_CVC_bipartite(G): solves the CVC problem on a bipartite graph G by applying Russian Doll Search.
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


def RDS_CVC(G, heur_start=True, time_limit = 3600):
    num_nodes = 0
    time_s = time.time()
    Best_val = 0
    Best_S = []

    # compute the cut-vertices of G, that will be excluded from the search
    C = list(articulation_points(G))

    G_bar = complement(G)

    if heur_start and G.number_of_nodes() > 3:  # Initializes best_val with heuristic solution
        Best_S = heuristic_solution(G)
        Best_val = len(Best_S)

    # sorts the vertices by degree
    vtx = sort_degree(G.subgraph(v for v in G.nodes() if v not in C).copy())

    vtx.reverse()

    # traverses the vertices in reverse order and starts a B&B search imposing that v_i is in S,
    # and U is restricted to v_{i+1}, ..., v_n
    A = []
    for i in range(len(vtx)):

        C = list(articulation_points(G.subgraph([u for u in G.nodes() if u != vtx[i]])))
        start_U = [v for v in vtx[i + 1:] if not G.has_edge(vtx[i], v) and v not in C]
        if len(start_U) + 1 <= Best_val:
            continue

        col = greedy_col_comp(G_bar.subgraph(start_U), True)

        A.append([[vtx[i]], start_U, col[0], col[1]])

        while A:
            [S, U, coloring, UB] = A.pop()

            len_S = len(S)

            while U and Best_val < len_S + UB:   # we keep exploring U until it can potentially improve on our current best solution
                num_nodes += 1
                v = max(U, key=lambda i: coloring[i])
                U.remove(v)

                if U:
                    # we remove v from the coloring and improve the upper bound if the color of v is not present anymore
                    c_v = coloring[v]
                    del coloring[v]
                    if c_v not in coloring.values():
                        UB_v = UB - 1
                    else:
                        UB_v = UB
                    A.append([S, U, coloring.copy(), UB_v])

                S = S + [v]
                len_S += 1
                if U:
                    C = list(articulation_points(G.subgraph([u for u in G.nodes() if u not in S])))
                    U = [u for u in U if not G.has_edge(u, v) and u not in C]
                    col = greedy_col_comp(G_bar.subgraph(U), False)
                    if col[1] < UB:
                        UB = col[1]
                        coloring = col[0]

                if Best_val < len_S:
                    Best_val = len_S
                    Best_S = S

        if time.time() - time_s >= time_limit:
            return [Best_val, time.time() - time_s, num_nodes, False]

    return [Best_val, Best_S, round(time.time() - time_s, 1), num_nodes, True]

# Computes a greedy coloring of graph H. Applies interchange strategy if greedy_int is set to True
def greedy_col_comp(H, greedy_int= False):
    col = greedy_color(H, interchange= greedy_int)
    num = len(set(col.values()))
    return [col, num]

def heuristic_solution(G, max_nsteps = 100, max_stab = []):
    if max_stab:
        S = list(max_stab)
    else:
        S = list(algorithms.approximation.clique.maximum_independent_set(G))

    comp = list(connected_components(G.subgraph([v for v in G.nodes() if v not in S])))
    #print('components: '+str(len(comp)))

    nsteps = 0
    while len(comp) > 1:
        nsteps += 1
        v = max(S, key = lambda v : len([c for c in comp if [u for u in G.neighbors(v) if u in c]]))
        S.remove(v)
        if nsteps < max_nsteps:
            for u in G.neighbors(v):
                if u not in S and not [w for w in G.neighbors(u) if w in S]:
                    S.append(u)


        comp = list(connected_components(G.subgraph([v for v in G.nodes() if v not in S])))
    #print('classic steps ' + str(nsteps))
    return S

# sort nodes such that v_i has maximum degree in G[v_i, ..., v_n] (minimum if reverse is False)
def sort_degree(G, reverse = True):
    nodes = []
    for _ in range(G.number_of_nodes()):
        if reverse:
            v = max(G.nodes(), key=G.degree())
        else:
            v = min(G.nodes(), key=G.degree())
        G.remove_node(v)
        nodes.append(v)
    return nodes

def RDS_CVC_bipartite(G, heur_start=True, time_limit = 3600):
    num_nodes = 0
    time_s = time.time()
    Best_val = 0
    Best_S = []

    # compute the cut-vertices of G, that will be excluded from the search
    C = articulation_points(G)

    # store the nodes of one bipartition of G
    top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}

    if heur_start: # Initializes best_val with heuristic solution
        Best_S = heuristic_solution(G, max_stab = get_max_stab_bipartite(G, top_nodes))
        Best_val = len(Best_S)

    # sorts the vertices by degree
    vtx = sort_degree(G.subgraph(v for v in G.nodes() if v not in C).copy())

    # traverses the vertices in reverse order and starts a B&B search imposing that v_i is in S,
    # and U is restricted to v_{i+1}, ..., v_n
    A = []
    for i in range(len(vtx))[::-1]:
        C = articulation_points(G.subgraph([u for u in G.nodes() if u != vtx[i]]))
        A.append([[vtx[i]], [v for v in vtx[i+1:] if not G.has_edge(vtx[i], v) and v not in C]])
        while A:
            [S, U] = A.pop()

            len_S = len(S)

            UB = get_alpha_bipartite(G.subgraph(U), {u for u in top_nodes if u in U})

            while U and Best_val < len_S + UB:  # we keep exploring U until it can potentially improve on our current best solution
                num_nodes += 1

                v = U.pop(0)
                if U:
                    A.append([S, U])

                S = S + [v]
                len_S += 1
                if Best_val < len_S:
                    Best_val = len_S
                    Best_S = S
                if U:
                    C = list(articulation_points(G.subgraph([u for u in G.nodes() if u not in S])))
                    U = [u for u in U if not G.has_edge(u, v) and u not in C]
                    UB = get_alpha_bipartite(G.subgraph(U), {u for u in top_nodes if u in U})

            if time.time() - time_s >= time_limit:
                return [Best_val, Best_S, time.time() - time_s, num_nodes, False]

    return [Best_val, Best_S, round(time.time() - time_s, 1), num_nodes, True]


# Returns the independence number of a bipartite graph
def get_alpha_bipartite(G, top_nodes):
    matching = bipartite.maximum_matching(G, top_nodes = top_nodes)

    return G.number_of_nodes() - len(matching.keys())/2

# Returns the maximum stable set of a bipartite graph
def get_max_stab_bipartite(G, top_nodes):
    matching = bipartite.maximum_matching(G, top_nodes=top_nodes)
    vertex_cover = bipartite.to_vertex_cover(G, matching, top_nodes=top_nodes)
    return [v for v in G.nodes() if v not in vertex_cover]