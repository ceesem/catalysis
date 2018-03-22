import numpy as np
import scipy as sp
import catalysis as cat
import sys
from copy import deepcopy

def transform_neuronlist(
                        nrns,
                        from_group=None,
                        to_group=None,
                        CatmaidInterface=None,
                        from_landmarks=None,
                        to_landmarks=None ):
    """

    """
    found_landmarks = False
    if from_landmarks is None:
        if to_landmarks is None:
            print("Requesting landmarks...")
            landmarks = landmark_group_list( CatmaidInterface )
            from_landmarks, to_landmarks = landmark_points( landmarks[from_group],
                                                            landmarks[to_group],
                                                            CatmaidInterface )
            found_landmarks = True
    elif to_landmarks is not None:
        print("Using stored landmarks...")
        found_landmarks = True
    else:
        raise ValueError('Either server info or landmarks must be specified')

    nrns_t = deepcopy( nrns )
    for nrn in nrns_t:
        nrns_t.neurons[nrn.id] = transform_neuron_from_landmarks(
                                    nrn,
                                    from_landmarks=from_landmarks,
                                    to_landmarks=to_landmarks)
    return nrns_t


def transform_neuron_from_landmarks(nrn,
                                    from_group=None,
                                    to_group=None,
                                    CatmaidInterface=None,
                                    from_landmarks=None,
                                    to_landmarks=None ):
    """
    Use moving least squares to transform all space-related points on a neuron,
    either by requesting points from the catmaid server or with stored
    local landmarks.
    """

    found_landmarks = False

    if from_landmarks is None:
        if to_landmarks is None:
            print("Requesting landmarks...")
            landmarks = landmark_group_list( CatmaidInterface )
            from_landmarks, to_landmarks = landmark_points( landmarks[from_group],
                                                            landmarks[to_group],
                                                            CatmaidInterface )
            found_landmarks = True
    elif to_landmarks is not None:
        print("Using stored landmarks...")
        found_landmarks = True
    else:
        raise ValueError('Either server info or landmarks must be specified')

    nrn_t = deepcopy(nrn)

    nrn_t.name = nrn_t.name + ' (Transformed)'
    nrn_t.id = nrn_t.id

    # Transform all the nodes.
    #print("Transforming nodes...")
    for nid in nrn_t.nodeloc:
        v = np.array(nrn_t.nodeloc[nid])
        ws = compute_weights( v, from_landmarks )
        nrn_t.nodeloc[nid] = moving_least_squares_affine_from_landmarks(
                                                            v,
                                                            from_landmarks,
                                                            to_landmarks,
                                                            ws )
    # Transform all the synapse locations.
    #print("Transforming synapses...")

    for vid in nrn_t.inputs.locs:
        v = np.array(nrn_t.inputs.locs[vid])
        ws = compute_weights( v, from_landmarks )
        nrn_t.inputs.locs[vid] = moving_least_squares_affine_from_landmarks(
                                                            v,
                                                            from_landmarks,
                                                            to_landmarks,
                                                            ws )
    for vid in nrn_t.outputs.locs:
        v = np.array(nrn_t.outputs.locs[vid])
        ws = compute_weights( v, from_landmarks )
        nrn_t.outputs.locs[vid] = moving_least_squares_affine_from_landmarks(
                                                            v,
                                                            from_landmarks,
                                                            to_landmarks,
                                                            ws )

    return nrn_t

def moving_least_squares_affine_from_landmarks( v, ps, qs, ws ):
    """
    Compute the 3d affine deformation minizing
    sum_i( w_i * ( A(p_i) - q_i )^2 ) based on Schaefer et al 2006

    Parameters
    ----------
    v : 1x3 numpy array
        Three-vector of the point to be transformed

    matches : List of tuples
        Each match is (p, q) where p and q are the     
    """
    p_hat, p_star = recenter_landmarks( ps, ws)
    q_hat, q_star  = recenter_landmarks( qs, ws)

    # f(v) = sum_j A_j * qhat_j + q_star, where
    # A_j = (v - p_star) * (sum_i phat^T_i w_i phat_i )^-1 w_j phat^T_j
    # which I'll break into X * Yinv * Z_j for the purpose of clarity
    # where X is 1 x 3, Y is 3 x 3, Z is 3 x 1

    A = np.zeros( len(ps) )
    X = [v - p_star]
    Y = np.zeros( (3,3) )
    for ind, p in enumerate(p_hat):
        Y += np.einsum('i,,j->ij',p,ws[ind],p)
    Yinv = np.linalg.inv(Y)

    for ind, p in enumerate(p_hat):
        Z = ws[ind] * p
        A[ind] = X @ Yinv @ Z

    v_transform = np.einsum('i,ij->j',A,q_hat) + q_star

    return v_transform


def recenter_landmarks( ps, ws ):
    """

    """
    p_star = np.einsum('i,ij->j',ws,ps) / np.sum(ws)
    return  ps - np.matlib.repmat(p_star,len(ps),1), p_star

def compute_weights( v, p, alpha = 2):
    """
    """
    ds = sp.spatial.distance.cdist(p,[v])
    return np.squeeze( np.power(1/( ds + sys.float_info.epsilon ), alpha) )

def landmark_group_list( CatmaidInterface ):
    """
    """
    groups_raw = CatmaidInterface.get_landmark_groups(
                                            with_locations=True,
                                            with_members=False )

    landmark_groups = {}    # Dict of group name->id
    for group in groups_raw:
        landmark_groups[ group['name'] ] = [ loc['id'] for loc in group['locations'] ]

    return landmark_groups

def landmark_points( from_group_ids, to_group_ids, CatmaidInterface ):
    """
        Find matched pairs of landmarks from two landmark groups.
    """
    landmarks = CatmaidInterface.get_landmarks( with_locations=True )
    
    from_points = []
    to_points = []

    for lm in landmarks:
        p = None
        q = None
        for loc in lm['locations']:
            if loc['id'] in from_group_ids:
                p = [ loc['x'], loc['y'], loc['z'] ]
            elif loc['id'] in to_group_ids:
                q = [ loc['x'], loc['y'], loc['z'] ]

        if q is not None and p is not None:
            from_points.append(p)
            to_points.append(q)

    return np.array( from_points ), np.array( to_points )




