import numpy as np
import hnswlib
import networkx as nx
from numba import njit
from numba.typed import List

M=200
ef_construction=200
ef=200

def get_kNN(X, q=15):
    """
    Generate a k-nearest neighbors graph from the input data.
    :param X: Input data (numpy array).
    :param q: Number of nearest neighbors.
    :return: k-nearest neighbors list and distances.
    """
    n = X.shape[0]
    dim = X.shape[1]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=n, ef_construction=200, M=64)
    p.add_items(X)
    p.set_ef(2*q)

    labels, dists = p.knn_query(X, k=q+1)
    knn_list = labels[:, 1:]
    knn_dists = dists[:, 1:]

    return knn_list, knn_dists


def init_random_walk(G,init_walk_len=10):
    n = G.number_of_nodes()
    v_cover=np.ones((n))

    hmap={}
    t=0
    for u in G.nodes():
        hmap[u]=t
        t+=1


    in_list = [[] for _ in range(n)]
    degs=np.zeros((n))
    for u in G.nodes():
        degs[hmap[u]]=G.out_degree(u)



    deg_list=[]
    for u in G.nodes():
        x=[]
        for v in G.predecessors(u):
            wt_uv=G.edges[v,u]['weight']
            x.append(wt_uv/degs[hmap[v]])
            in_list[hmap[u]].append(hmap[v])
        deg_list.append(np.array(x))


    #Now change v_cover:
    for ell in range(init_walk_len):

        v_cover_n=np.zeros((n))
        for i in range(n):
            v_cover_n[i]=sum(deg_list[i]*v_cover[in_list[i]])

        v_cover=v_cover_n.copy()

    init_score={}

    #Add init_score back to the graph
    for u in G.nodes():
        init_score[u]=np.float32(v_cover[hmap[u]])

    return init_score


@njit
def ascending_walk(ng_list,init_score,times):



    n0=len(init_score)
    v_cover=np.zeros((n0))

    walker=[]
    walker_ng_list=[]

    for ell in range(n0):
        x=[]
        for v in ng_list[ell]:
            if init_score[v]>init_score[ell]:
                x.append(v)


        walker.append(x)
        walker_ng_list.append(len(x))



    v_cover_n=np.zeros((n0))

    for rounds in range(times):

        #print("rounds=",rounds)

        v_cover=np.zeros((n0))

        for u in range(n0):
            v=u
            while walker_ng_list[v]>0:
                v= walker[v][np.random.randint(0, walker_ng_list[v])]

            v_cover[u]=init_score[v]

        v_cover_n=v_cover_n+v_cover


    v_cover_n=v_cover_n/times

    return v_cover_n



def FLOW_rank_optimized(X,init_score,r):

    times=200

    knn_list, _=get_kNN(X, r)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)

    n=X.shape[0]



    ng_list=[]

    #Preparing for numba. ng_list is a n*r matrix.
    for ell in range(n):
      x=[]
      for v in G.neighbors(ell):
        x.append(int(v))

      ng_list.append(List(x))

    init_score_numba=np.zeros((n))
    for u in init_score:
        init_score_numba[u]=init_score[u]

    v_cover_n=ascending_walk(ng_list,init_score_numba,times)

    final_score={}
    for u in G.nodes():
        final_score[u]=init_score[u]/(v_cover_n[u]+0.00001)




    return final_score


def FlowRank(X,q=20,r=20):


    #Get initial density estimation.

    knn_list, _=get_kNN(X, q)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)


    init_score=init_random_walk(G)

    final_score=FLOW_rank_optimized(X,init_score,r)


    return final_score


import importlib
from numba import njit


# We obtain t nearest neighbors to each of the current partitions, and then use this to obtain the final layer-i to layer-i-1 edges. This is to deal with weakness of ANN algorithms in dealing with OOD queries.

def CDNN_layer(X, nodes_this_round, final_labels, t):
    all_neighbors = []
    all_distances = []

    # Build current_partition

    true_k = len(set(final_labels)) - 1
    n = len(final_labels)
    current_partition = [[] for _ in range(true_k)]
    for ell in range(n):
        val = final_labels[ell]
        if val != -1:
            current_partition[val].append(ell)

    # done

    dim = X.shape[1]
    queries = X[nodes_this_round]

    for cluster_nodes in current_partition:
        if len(cluster_nodes) == 0:
            continue

        # 1) Extract data for this cluster
        cluster_data = X[cluster_nodes]

        # 2) Build an HNSW index on the fly
        p = hnswlib.Index(space='l2', dim=dim)
        p.init_index(max_elements=len(cluster_nodes),
                     ef_construction=ef_construction,
                     M=M)

        # 3) Add items, using the *actual* indices as labels
        labels = np.array(cluster_nodes, dtype=np.int64)
        p.add_items(cluster_data, labels)

        # 4) Set query‐time ef
        p.set_ef(ef)

        # 5) Query all rem_nodes at once
        k_ = min(2 * t, len(cluster_nodes))
        # print('len cluster_nodes:', len(cluster_nodes), 'k_:', k_)
        nbrs, dists = p.knn_query(queries, k=k_)

        all_neighbors.append(nbrs)
        all_distances.append(dists)

    # 6) Stack results from every cluster: shape = (Q, C·t)
    all_neighbors = np.concatenate(all_neighbors, axis=1)
    all_distances = np.concatenate(all_distances, axis=1)

    # 7) For each query, pick the top-t closest across *all* clusters
    idx_sort = np.argsort(all_distances, axis=1)[:, :t]

    Q = len(nodes_this_round)
    knn_list = np.zeros((Q, t), dtype=np.int64)
    knn_dists = np.zeros((Q, t), dtype=all_distances.dtype)

    for i in range(Q):
        sel = idx_sort[i]
        knn_list[i] = all_neighbors[i, sel]
        knn_dists[i] = all_distances[i, sel]

    return knn_list, knn_dists


def ng_heat_kernel(distances):
    n_points = len(distances)
    sigmas = np.zeros(n_points)
    target = np.zeros(n_points)
    rhos = np.zeros(n_points)

    P_vec = [[] for _ in range(n_points)]

    # Step 1: Compute rho_i (local connectivity)
    for i in range(n_points):

        rhos[i] = distances[i][0] if len(distances[i]) > 0 else 0  # Minimum nonzero distance

        if rhos[i] != min(distances[i]):
            raise KeyError(f"0-th index is not closest, {rhos[i]:.3f} {min(distances[i]):.3f}")

    # Step 2: Solve for sigma_i using binary search
    def find_sigma(i, target_i):

        lo, hi = 1e-8, 100000.0  # Search range for sigma
        for _ in range(64):  # Binary search
            sigma = (lo + hi) / 2.0
            weights = np.exp(-(distances[i] - rhos[i]) / sigma)
            if np.sum(weights) > target_i:
                hi = sigma
            else:
                lo = sigma
        return (lo + hi) / 2.0

    for i in range(n_points):
        if len(distances[i]) > 0:
            c1 = len(distances[i])

            if c1 == 1:
                target[i] = 1

            elif c1 <= 5:
                target[i] = 1 + c1 / 2

            else:
                target[i] = np.log2(c1)

            sigmas[i] = find_sigma(i, target[i])

    # Step 3: Compute edge weights
    for i in range(n_points):
        for d in distances[i]:
            weight = np.exp(-(d - rhos[i]) / sigmas[i])

            P_vec[i].append(weight)

    P_vec = np.array(P_vec)

    assert np.shape(P_vec) == np.shape(distances)

    for i in range(n_points):
        P_vec[i] = P_vec[i] / sum(P_vec[i])

    return P_vec


@njit
def pseudo_centroid_distances(labels, nodes_this_round, P_vec, ng_list, dist_to_centroids=None, node_to_rank=None):
    n0, k0 = np.shape(ng_list)
    true_k_temp = len(dist_to_centroids[0])

    for i in range(n0):

        u = nodes_this_round[i]

        dists_temp = np.zeros(true_k_temp, dtype=np.float64)
        for j in range(k0):
            v = ng_list[i, j]

            for ell in range(true_k_temp):
                # print(ell)
                # print(v)
                # print(node_to_rank[v])
                # print(dist_to_centroids[node_to_rank[v]])

                dists_temp[ell] += P_vec[i, j] * dist_to_centroids[node_to_rank[v]][ell]

        #            dists_temp+=P_vec[i,j]*dist_to_centroids[node_to_rank[v]]

        c_idx = np.argmin(dists_temp)
        labels[u] = c_idx

        dist_to_centroids.append(dists_temp)

    return labels, dist_to_centroids


from sklearn.cluster import KMeans
import time


def MCPC_Kmeans(X, true_k, q=20, r=20, t=20, layer_ratio=None, scores=None, choose_min_obj=True,
                send_meta_data=False):
    if layer_ratio is not None:
        top_frac = layer_ratio[0]

    else:
        raise KeyError("The C-P partitions cannot be None")

    num_step = len(layer_ratio) - 1
    n = X.shape[0]

    t0 = time.time()
    # Get the ranking score.
    if scores is None:
        scores = FlowRank(X, q, r)

    # print(f"FlowRank={time.time()-t0:.3f}")

    # Get top frac fractions points
    sorted_points = np.array(sorted(scores, key=scores.get, reverse=True)).astype(int)
    core_nodes = sorted_points[0:int(top_frac * n)]

    # Apply K-Means on cores
    X_core = X[core_nodes]

    if choose_min_obj:
        min_obj_val = float('inf')

        for rounds in range(20):

            kmeans = KMeans(n_clusters=true_k, n_init=1, max_iter=1000)
            kmeans.fit(X_core)

            centroids = kmeans.cluster_centers_
            obj_val = kmeans.inertia_
            labels_km = kmeans.labels_

            if rounds == 0 or obj_val < min_obj_val:
                min_obj_val = obj_val
                best_centroids = centroids
                best_labels_km = labels_km

        centroids = best_centroids
        labels_km = best_labels_km

    else:
        kmeans = KMeans(n_clusters=true_k)
        kmeans.fit(X_core)
        centroids = kmeans.cluster_centers_
        labels_km = kmeans.labels_

    # Start generating final labels
    final_labels = -1 * np.ones(n)
    final_labels[core_nodes] = labels_km
    final_labels = final_labels.astype(int)

    # For each point, get the (pseudo) distances to all centroids
    # stored in the order of reverse sorted according to scores.
    dist_to_centroids = []
    for core_node in core_nodes:
        cen_dist = []
        for center_idx, center in enumerate(centroids):
            cen_dist.append(np.linalg.norm(X[core_node] - center))
        ###dist_to_centroids.append(np.array(cen_dist).astype(float))

        dist_to_centroids.append(np.array(cen_dist).astype('float64'))

    # We will create the CDNN graph layer-wise

    #    -------------

    # Rank of nodes according to reversed score.
    node_to_rank = -1 * np.zeros(n).astype(int)
    rank_counter = 0
    for node in core_nodes:
        node_to_rank[node] = rank_counter
        rank_counter += 1

    # Current_partition is used only for calculating layer-wise CDNN

    # Find nearest centroid layer by layer for periphery nodes
    for rnd in range(num_step):

        nodes_this_round = sorted_points[int(layer_ratio[rnd] * n):int(layer_ratio[rnd + 1] * n)].astype(int)

        # We create the CDNN graph layer-by-layer
        ng_list, distances = CDNN_layer(X, nodes_this_round, final_labels, t)

        # Get the delegation weights.
        P_vec = ng_heat_kernel(distances)

        # Obtain the final cluster allocation through pseudo distance calculation
        final_labels, dist_to_centroids = pseudo_centroid_distances(final_labels, nodes_this_round, P_vec, ng_list,
                                                                    dist_to_centroids, node_to_rank)

        for peri_node in nodes_this_round:
            node_to_rank[peri_node] = rank_counter
            rank_counter += 1

    if send_meta_data:
        return final_labels, obj_val, centroids

    return final_labels

