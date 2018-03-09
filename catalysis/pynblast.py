import pandas as pd
import scipy as sp
import numpy as np
from bisect import bisect
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial

def nblast_neuron_pair( nrn_q, nrn_t, smat, resample_distance = 1000, num_nn = 5 ):
    """
        Find the NBLAST score for two neurons, given a score matrix object
    """
    d_and_udotv = neuron_comparison_nblast_components( nrn_q, nrn_t, resample_distance=resample_distance, num_nn = num_nn )
    S = nblast_dist_fun( d_and_udotv, smat )
    return S

def _nblast_neuron_pair_for_mp( input ):
    """
        Find the NBLAST score for two neurons, given a score matrix. Only takes one argument so multiprocessing map can be used.
    """
    nrn_q_dotprop = input[0]
    nrn_t_dotprop = input[1]
    smat = input[2]
    resample_distance = input[3]
    num_nn = input[4]
    d_and_udotv = neuron_comparison_nblast_components( nrn_q_dotprop, nrn_t_dotprop, resample_distance=resample_distance, num_nn = num_nn, as_dotprop = True )
    S = nblast_dist_fun( d_and_udotv, smat )
    return S

def neuron_to_dotprop( nrn, resample_distance, num_nn=5 ):
    """
        Convert a neuron to a dotprop cloud for NBLAST comparison.
    """
    pths = nrn.minimal_paths()
    dp = []
    for ind, pth in enumerate(pths):
        dp.append( dotprop_path( [nrn.nodeloc[nid] for nid in pth],
                          resample_distance=resample_distance,
                         num_nn=num_nn ) )
    return np.vstack(dp)

def dotprop_path( xyz, resample_distance, num_nn=5 ):
    """
        Take a set of xyz coordinates for an ordered path in Euclidean space and return an interpolated dotprop representation.
    """

    xyzi = interpolate_path( xyz, resample_distance=resample_distance)
    dist_tree = sp.spatial.KDTree( xyzi )
    v = []
    query_nn = np.min( (num_nn+1, len(xyzi)) )
    for n in xyzi:
        nnxyz = xyzi[ dist_tree.query( n, query_nn  )[1] ]  # +1 because this will return original node also.
        nnxyz_c = nnxyz - np.matlib.repmat( np.mean(nnxyz, axis=0), len(nnxyz), 1 )
        U,s,Vh = np.linalg.svd(nnxyz_c)
        v.append(Vh[0])
    v = np.array(v)
    return np.concatenate( (xyzi, v), axis=1 )

def interpolate_path( xyz, resample_distance ):
    """
        Resample a path along a neuron. Discards neuron-level topological information, so would have to be applied to slabs to preserve branch points.
    """
    xs = [x[0] for x in xyz]
    ys = [x[1] for x in xyz]
    zs = [x[2] for x in xyz]

    lens = [np.linalg.norm(u) for u in np.diff(xyz, axis=0)]
    cumlen = np.insert( np.cumsum(lens ), 0, 0 )

    parts = np.modf( cumlen[-1] / resample_distance )
    if parts[1] < 1:
        nni = 1
    elif parts[0] < 0.5 and parts[1] > 0:
        nni = parts[1]
    else:
        nni = parts[1]+1
    li = np.linspace(0,cumlen[-1],int(nni))

    xi = np.interp(li, cumlen, xs)
    yi = np.interp(li, cumlen, ys)
    zi = np.interp(li, cumlen, zs)

    return np.array( [ [xi[ind], yi[ind], zi[ind]] for ind,val in enumerate(xi)] )

def neuron_comparison_nblast_components( source_nrn, target_nrn, resample_distance=1000, num_nn=5, as_dotprop = False ):
    """
        Compute the distance and dot product values comparing the nodes in the source neuron to those in the target neuron.
    """

    if as_dotprop:
        source_dp = source_nrn
        target_dp = target_nrn
    else:
        source_dp = neuron_to_dotprop( source_nrn, resample_distance=resample_distance, num_nn=num_nn)
        target_dp = neuron_to_dotprop( target_nrn, resample_distance=resample_distance, num_nn=num_nn)
    udotv = []
    dist_tree = sp.spatial.KDTree( target_dp[:,0:3] )
    ds = dist_tree.query( source_dp[:,0:3], 1 )
    # for node in source_dp:
    #     targ_node_info = dist_tree.query( node[0:2], 1 )
    #     d.append( targ_node_info[0] )
    #     udotv.append( np.abs( np.dot(node[3:6], target_dp[targ_node_info[1]][3:6]) ) )
    d_and_udotv = np.zeros( (len(ds[0]),2) )
    d_and_udotv[:,0] = ds[0]
    d_and_udotv[:,1] = np.einsum('ij,ij->i', source_dp[:,3:], target_dp[ds[1],3:])
    return d_and_udotv

def nblast_dist_fun( d_and_udotv, score_lookup ):
    S = 0
    if np.shape(d_and_udotv)[1] != 2:
        raise ValueError( "Scores must only have two components")
    for row in d_and_udotv:
        S += score_lookup.score( d=row[0], udotv=row[1] )
    return S

class ScoreMatrixLookup:
    def __init__( self, mat, d_range, udotv_range):
        self.mat = np.array( mat )
        self.d_range = self.process_range_values( d_range )
        self.udotv_range = self.process_range_values( udotv_range )

    @classmethod
    def from_dataframe( cls, df ):
        mat = df.values
        d_range = list(df.index)
        udotv_range = list(df.columns)
        return cls( mat, d_range, udotv_range )

    def process_range_values( self, range_list ):
        val_range = []
        for row in range_list:
            val_array = [ float(num) for num in row[1:-1].split(',') ]
            val_range.append( val_array[0] )
        val_range.append( np.Inf )
        return np.array(val_range[1:-1])

    def score( self, d, udotv ):
        ind_d = bisect( self.d_range, d )
        ind_udotv = bisect( self.udotv_range, udotv )
        return self.mat[ind_d,ind_udotv]


def nblast_neurons(smat, nrns_q, nrns_t=None, resample_distance=1000, num_nn=5, min_strahler=None, processes=4 ):
    """
        Query a list of neurons against one another and return as a data frame. Uses multiprocessing by default.
    """
    if type(nrns_q) is 'NeuronObj':
        nrns_q = [nrns_q]

    nrns_q_dotprop = {}

    pool = Pool( processes=processes )
    neuron_to_dotprop_cond = partial( neuron_to_dotprop, resample_distance=resample_distance, num_nn=num_nn)
    if min_strahler is None:
        nrns_q_dotprop_list = pool.map( neuron_to_dotprop_cond, nrns_q )
    else:
        nrns_q_dotprop_list = pool.map( neuron_to_dotprop_cond, [nrn.strahler_filter(min_strahler) for nrn in nrns_q] )
    nrns_q_dotprop = {nrn.id:nrns_q_dotprop_list[ii] for ii, nrn in enumerate(nrns_q) }

    if nrns_t is None:
        nrns_t = nrns_q
        nrns_t_dotprop = nrns_q_dotprop
    else:
        nrns_t_dotprop = {}
        for nrn in nrns_t:
            if min_strahler is None:
                nrns_t_dotprop_list = pool.map( neuron_to_dotprop_cond, nrns_t )
            else:
                nrns_t_dotprop_list = pool.map( neuron_to_dotprop_cond, [nrn.strahler_filter(min_strahler) for nrn in nrns_t] )
            nrns_t_dotprop = {nrn.id:nrns_t_dotprop_list[ii] for ii, nrn in enumerate(nrns_t) }

    queries = []
    targets = []
    similarities = []
    args = []

    for nrn_q in nrns_q:
        q_name = nrn_q.name + ' (' + str(nrn_q.id) + ')'
        for nrn_t in nrns_t:
            t_name = nrn_t.name + ' (' + str(nrn_t.id) + ')'
            queries.append(q_name)
            targets.append(t_name)
            args.append((nrns_q_dotprop[nrn_q.id], nrns_t_dotprop[nrn_t.id], smat, resample_distance, num_nn) )

    similarities = pool.map( _nblast_neuron_pair_for_mp, args  )
    df = pd.DataFrame({'Queries':queries, 'Targets':targets, 'S':similarities}).reindex(columns=['Queries','Targets','S'])
    return df.pivot(index='Queries',columns='Targets',values='S')
