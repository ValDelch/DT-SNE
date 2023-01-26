# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

# This code is implemented based on the code of Laurens van der Maaten and Adrien Bibal
# for DT-TSNE and based on the code of Géraldin Nanfack & Valentin Delchevalerie for DTs.
# Parts of the code using the implementation of Laurens van der Maaten are highlighted.
# It implements a faster DT-TSNE algorithm using cython

# Libraries & setup
import numpy as np
cimport numpy as np
np.import_array()

import networkx as nx

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_intp SIZE_t


# Functions


# Function from the python t-SNE implementation of Laurens van der Maaten without any modification
# https://lvdmaaten.github.io/tsne/
def Hbeta(D = np.array([]), beta = 1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = np.maximum(sum(P), EPSILON)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


# Function from the python t-SNE implementation of Laurens van der Maaten without any modification
# https://lvdmaaten.github.io/tsne/
def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax =  np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while (np.isnan(Hdiff) or np.abs(Hdiff) > tol) and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return P



def compute_pij(X, perplexity=15):
    # Compute pij (from the code of vdm)
    pij = x2p(X=X, perplexity=perplexity)
    pij = pij + np.transpose(pij)
    pij = pij / np.sum(pij)
    pij = np.maximum(pij, 1e-12) # In order to avoid having zeros in the matrix
    return pij



cpdef dt_sne(np.ndarray[DOUBLE_t, ndim=2] X, list feature_names, double perplexity, double rs_rate, int min_leaves):

    cdef:
        SIZE_t n = np.shape(X)[0]
        SIZE_t d = np.shape(X)[1]
        SIZE_t node, feature, end, p, other_instance, left_instance, right_instance, instance, max_samples

        double[:,:] pij = compute_pij(X=X, perplexity=perplexity)
        np.ndarray[DOUBLE_t, ndim=2] actual_qij = np.ones((n,n))
        np.ndarray[DOUBLE_t, ndim=2] d_other_qij, d_lr_qij
        SIZE_t[:] sorted_indices, left_instances, right_instances, _best_left, _best_right, best_left, best_right, indices
        DOUBLE_t[:] X_feat
        np.ndarray[DOUBLE_t, ndim=2] X_in_node
        np.ndarray[DOUBLE_t, ndim=2] Q
        np.ndarray[SIZE_t, ndim=1] in_node, in_node_sorted

        object G = nx.DiGraph()

        list splittable_nodes = [1] # the root is the only node to expand yet
        list leaves = []
        int counter = 2 # Counter is set at 2 because the next ID will be 2

        SIZE_t best_node, _best_node
        SIZE_t best_feature, _best_feature
        double KL_score, best_threshold, _best_threshold, current_score, _best_score
        double best_score
        double[:,:] best_qij, _best_qij

    # Init the tree
    G.add_node(1, instances=list(range(n)), attribute="") # Add the ID of all instances in the root node
    
    best_score = INFINITY

    # Loop untill the end
    while len(splittable_nodes) != 0:

        print(len(splittable_nodes))

        current_score = best_score

        # Try to split on node "node"
        for node in splittable_nodes:

            _best_score = current_score

            # Get the instances in the node
            in_node = np.array(G.nodes[node]['instances']).astype(np.intp)
            X_in_node = X[in_node[:], :]
            end = X_in_node.shape[0]

            # Check that this node is indeed splittable (n > 1)
            if not end > 2*min_leaves:
                splittable_nodes.remove(node)
                leaves.append(node)
                continue
            elif rs_rate != 0:
                max_samples = int(np.maximum(end * rs_rate, 1))
                indices = np.random.choice(end-1, size=max_samples, replace=False).astype(np.intp)

            # Try to split on the feature "feature"
            for feature in range(d):

                # Sort the instances w.r.t. the feature's value
                sorted_indices = np.argsort(X_in_node[:,feature])
                X_feat = X_in_node[sorted_indices, feature]
                in_node_sorted = in_node[sorted_indices]

                # No variance on this feature, so skip it
                if X_feat[end-1] <= X_feat[0] + EPSILON:
                    continue

                #Loop to search for the best threshold
                for p in range(min_leaves - 1, end-min_leaves-1):
                    if X_feat[p+1] <= X_feat[p] + EPSILON:
                        continue
                    elif (rs_rate != 0) and (p not in indices):
                        continue

                    # Get the groups
                    left_instances = in_node_sorted[0:p+1]
                    right_instances = in_node_sorted[p+1:]

                    # Reset d_qij to zero
                    d_other_qij = np.zeros((n,n))
                    d_lr_qij = np.zeros((n,n))

                    # Add a distance for all instances that are separated
                    for other_instance in range(n):

                        # Is other_instance in left or right ?
                        in_left = False ; in_right = False
                        if other_instance in left_instances:
                            in_left = True
                        elif other_instance in right_instances:
                            in_right = True

                        # If other_instance is in right, we update only d_lr_qij
                        if in_right:
                            for left_instance in left_instances:
                                d_lr_qij[other_instance, left_instance] += 1
                                d_lr_qij[left_instance, other_instance] += 1

                    d_lr_qij[range(n),range(n)] = 0. ; d_other_qij[range(n),range(n)] = 0.

                    Q = 1. - ((actual_qij + d_other_qij + d_lr_qij) / np.maximum(np.max(actual_qij + d_other_qij + d_lr_qij), EPSILON))
                    Q[range(n),range(n)] = 0.
                    Q = Q / np.maximum(np.sum(Q), EPSILON)
                    Q = np.maximum(Q, EPSILON)

                    #KL_score = np.sum(pij * np.log(pij / (1. - (qij / np.sum(qij)))))
                    KL_score = np.sum(pij * np.log(pij / Q))

                    # Save the best split that we found until now
                    if _best_score > KL_score:

                        _best_left = np.copy(in_node_sorted[0:p+1])
                        _best_right = np.copy(in_node_sorted[p+1:])

                        _best_score = KL_score
                        _best_node = node
                        _best_feature = feature
                        _best_threshold = (X_feat[p] + X_feat[p+1]) / 2.
                        _best_qij = (actual_qij + d_other_qij + d_lr_qij)


            # If none of the splits for the node leads to a smaller KL, then we can stop trying to split it anymore
            #if not current_score > _best_score:

                #print('Node', node, 'is not splittable ')
                #splittable_nodes.remove(node)
                #leaves.append(node)

            # Save the best split (the one that reduces the most the KL)
            if _best_score < best_score:

                best_left = np.copy(_best_left)
                best_right = np.copy(_best_right)

                best_score = _best_score
                best_node = _best_node
                best_feature = _best_feature
                best_threshold = _best_threshold
                best_qij = np.copy(_best_qij)

        # Perform the split if it exists
        if best_score < current_score:

            splittable_nodes.remove(best_node)

            actual_qij = np.copy(best_qij)

            # Update the graph
            G.add_node(counter, instances=best_left, attribute="", value="<= "+str(best_threshold)) # Add new node for the best split with the left instances
            G.add_edge(best_node, counter) # Add an edge between the parent and this node
            counter += 1 # Increase the counter for the next new node

            G.add_node(counter, instances=best_right, attribute="",value="> "+str(best_threshold))  # Add new node for the best split with the right instances
            G.add_edge(best_node, counter) # Add an edge between the parent and this node
            counter += 1 # Increase the counter for the next new node

            G._node[best_node]["attribute"] = feature_names[best_feature]

            # And add the two new leaves in the list of nodes that are subject to split
            splittable_nodes.append(counter-2)
            splittable_nodes.append(counter-1)

        else:
            for node in splittable_nodes:
                leaves.append(node)
            break

    return G, leaves
