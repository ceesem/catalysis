import numpy as np
import scipy as sp
import catalysis as cat
import sys
import tqdm
from copy import deepcopy

def transform_neuronlist(
                        nrns,
                        from_group=None,
                        to_group=None,
                        CatmaidInterface=None,
                        from_landmarks=None,
                        to_landmarks=None,
                        contralateral=True ):
    """

    """
    from_landmarks, to_landmarks = _parse_landmarks(from_group=from_group,
                                                    to_group=to_group,
                                                    CatmaidInterface=CatmaidInterface,
                                                    from_landmarks=from_landmarks,
                                                    to_landmarks=to_landmarks,
                                                    contralateral=contralateral)

    nrns_t = deepcopy( nrns )
    for nrn in tqdm.tqdm(nrns_t):
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
                                    to_landmarks=None,
                                    contralateral=True,
                                    transform_synapses=False ):
    """
    Use moving least squares to transform all space-related points on a neuron,
    either by requesting points from the catmaid server or with stored
    local landmarks.
    """

    from_landmarks, to_landmarks = _parse_landmarks(from_group=from_group,
                                                    to_group=to_group,
                                                    CatmaidInterface=CatmaidInterface,
                                                    from_landmarks=from_landmarks,
                                                    to_landmarks=to_landmarks,
                                                    contralateral=True)

    nrn_t = deepcopy(nrn)

    nrn_t.name = nrn_t.name + ' (Transformed)'
    nrn_t.id = nrn_t.id

    # Transform all the nodes.
    vs = np.array([nrn_t.nodeloc[nid] for nid in nrn_t.nodeloc])
    vprimes = moving_least_squares_affine_vectorized(vs,from_landmarks,to_landmarks)
    for ind, nid in enumerate(nrn_t.nodeloc):
        nrn_t.nodeloc[nid] = vprimes[ind,:]

    if transform_synapses:
        vins = np.array([nrn_t.inputs.locs[cid] for cid in nrn_t.inputs.locs])
        vinprimes = moving_least_squares_affine_vectorized( vins,
                                                            from_landmarks,
                                                            to_landmarks)
        for ind, cid in enumerate(nrn_t.inputs.locs):
            nrn_t.inputs.locs[cid] = vinprimes[ind,:]

        vouts = np.array([nrn_t.outputs.locs[cid] for cid in nrn_t.outputs.locs])
        voutprimes = moving_least_squares_affine_vectorized( vouts,
                                                            from_landmarks,
                                                            to_landmarks)
        for ind, cid in enumerate(nrn_t.outputs.locs):
            nrn_t.outputs.locs[cid] = voutprimes[ind,:]

    return nrn_t

def moving_least_squares_affine_from_landmarks( v, ps, qs, ws ):
    """
    Compute the 3d affine deformation minimizing
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

def moving_least_squares_affine_vectorized( vs, ps, qs ):
    """
    Compute the 3d affine deformation based on Schaefer et al 2006.
    Parameters
    ----------
    vs : Npoint x 3 numpy array
        Block of points to be transformed

    ps : Nlandmark x 3 numpy array
        Block of landmarks in the starting coordinates

    qs : Nlandmark x 3 numpy array
        Block of landmarks in the transformed coordinates, matched indices.
    
    Returns
    -------

    vprime : Npoint x 3 numpy array
        Block of points in the transformed coordinates.
    """
    Nlandmark = len(ps)
    Npoint = len( vs )

    # Reshape into arrays with consistent indices
    ps = ps.ravel().reshape(1,3,Nlandmark,1,order='F')
    qs = qs.ravel().reshape(1,3,Nlandmark,1,order='F')
    vs = vs.ravel().reshape(1,3,1,Npoint,order='F')

    ds = sp.spatial.distance.cdist( ps.ravel('F').reshape(Nlandmark,3),
                                    vs.ravel('F').reshape(Npoint,3),
                                    "sqeuclidean").reshape(1,1,Nlandmark,Npoint ) + sys.float_info.epsilon
    ws = 1/ds

    wi_norm_inv = 1/np.sum(ws,axis=2)
    pstar = np.einsum('ijl,ijkl,ijkl->ijl',wi_norm_inv,ws,ps).reshape(1,3,1,Npoint)
    qstar = np.einsum('ijl,ijkl,ijkl->ijl',wi_norm_inv,ws,qs).reshape(1,3,1,Npoint)

    phat = ps - pstar
    qhat = qs - qstar

    vminusp = vs - pstar

    Y = np.einsum('ijkl,mikl,mjkl->ijl',ws,phat,phat).reshape(3,3,1,Npoint)
    Yinv = np.zeros((3,3,1,Npoint))
    for i in range(Npoint):
        Yinv[:,:,0,i] = np.linalg.inv( Y[:,:,0,i] )
        
    Z = np.einsum('ijkl,mikl,mjkl->ijl',ws,phat,qhat).reshape(3,3,1,Npoint)

    vprime = np.einsum('iakl,abkl,bjkl->ijkl',vminusp,Yinv,Z) + qstar
    vprime = vprime.ravel('F').reshape(Npoint,3)
    return vprime


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

def url_to_transformed_point( xyz0,
                            CatmaidInterface,
                            zoomlevel=0,
                            from_group=None,
                            to_group=None,
                            from_landmarks=None,
                            to_landmarks=None,
                            contralateral=True):
    """
        
    """

    from_landmarks, to_landmarks = _parse_landmarks(from_group=from_group,
                                                    to_group=to_group,
                                                    CatmaidInterface=CatmaidInterface,
                                                    from_landmarks=from_landmarks,
                                                    to_landmarks=to_landmarks,
                                                    contralateral=contralateral)

    xyzt = moving_least_squares_affine_from_landmarks( xyz0,
                                                       from_landmarks,
                                                       to_landmarks,
                                                       ws )

    return CatmaidInterface.get_catmaid_url_to_point(xyzt, zoomlevel=zoomlevel)

def _parse_landmarks(from_group=None,
                     to_group=None,
                     CatmaidInterface=None,
                     from_landmarks=None,
                     to_landmarks=None,
                     contralateral=True):
    """
        Either takes landmarks as inputs or pulls them from a catmaid server.
        If contralateral is True (the default), all landmarks are used and mirrored so that
        a contralateral landmark is mapped to an ipsilateral point, etc.
    """
    if from_group is not None and to_group is not None and CatmaidInterface is not None:
        #print("Requesting landmarks...")
        landmarks = landmark_group_list( CatmaidInterface )
        from_landmarks, to_landmarks = landmark_points( landmarks[from_group],
                                                        landmarks[to_group],
                                                        CatmaidInterface )
    elif to_landmarks is None or to_landmarks is None:
        raise ValueError('Either complete server info or landmarks must be specified')

    if contralateral:
        store_to_landmarks = deepcopy(to_landmarks)
        to_landmarks = np.vstack((to_landmarks,from_landmarks))
        from_landmarks = np.vstack((from_landmarks,store_to_landmarks))

    return from_landmarks, to_landmarks

