import pandas as pd
import scipy as sp
import numpy as np
import networkx as nx
import re
import copy
from bisect import bisect
from multiprocessing import Pool
from functools import partial

def nblast_neuron_pair( nrn_q, nrn_t, score_lookup, resample_distance = 1000, num_nn = 5, normalize=False ):
    """
        Find the NBLAST score for two neurons, given a score matrix object
    """
    d_and_udotv = neuron_comparison_nblast_components( nrn_q, nrn_t, resample_distance=resample_distance, num_nn = num_nn )
    S = nblast_dist_fun( d_and_udotv, score_lookup )
    if normalize:
        S = S / max_blast_score( len(nrn_q.nodeloc), score_lookup )
    return S

def max_blast_score( L, score_lookup ):
    """
        Returns the maximum blast similarity for a dotprop of node length L, the max possible value of nblast for a given query, where all distances are 0 and all dot products are 1.
    """
    d_and_udotv = np.transpose( np.vstack( (np.zeros( L ),
                              np.ones( L )) ) )
    return nblast_dist_fun( d_and_udotv, score_lookup )

def _nblast_neuron_pair_for_mp( input ):
    """
        Find the NBLAST score for two neurons, given a score matrix. Only takes one argument so multiprocessing map can be used.
    """
    nrn_q_dotprop = input[0]
    nrn_t_dotprop = input[1]
    score_lookup = input[2]
    resample_distance = input[3]
    num_nn = input[4]
    normalize = input[5]

    # Check if neurons are effectively emtpy, likely do to Strahler pruning.
    if len( nrn_q_dotprop ) > 1 and len(nrn_t_dotprop) > 1:
        d_and_udotv = neuron_comparison_nblast_components( nrn_q_dotprop, nrn_t_dotprop, resample_distance=resample_distance, num_nn = num_nn, as_dotprop = True )
        S = nblast_dist_fun( d_and_udotv, score_lookup )
        
        if normalize:
            S = S / max_blast_score( len(nrn_t_dotprop), score_lookup )
    else:
        S = 0
    return S

def neuron_to_dotprop( nrn, resample_distance=1000, num_nn=5, min_strahler=None ):
    """
        Convert a neuron to a dotprop cloud for NBLAST comparison.
    """
    pths = nrn.minimal_paths()

    if min_strahler is not None:
        sn = nrn.strahler_number()
        if min_strahler > np.max( list( sn.values() ) ):
            return(np.zeros((1,6)) * np.nan)


        if min_strahler>0:
            for ind, pth in enumerate(pths):
                pths[ind] = [nid for nid in pth if sn[nid]>=min_strahler]
        else:
            min_strahler = np.max( list( sn.values() ) ) + min_strahler
            for ind, pth in enumerate(pths):
                pths[ind] = [nid for nid in pth if sn[nid]>=min_strahler]

    dp = []
    for ind, pth in enumerate(pths):
        if len(pth) > 0:
            dp.append( dotprop_path( [nrn.nodeloc[nid] for nid in pth],
                                  resample_distance=resample_distance,
                                 num_nn=num_nn ) )
    if len(dp) > 0:
        return np.vstack(dp)
    else:
        return np.nan * np.zeros((1,6))

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


def nblast_neurons(score_lookup, nrns_q, nrns_t=None, resample_distance=1000, num_nn=5, min_strahler=None, normalize=False, processes=4 ):
    """
        Query a list of neurons against one another and return as a data frame. Uses multiprocessing by default.
    """
    if type(nrns_q) is 'NeuronObj':
        nrns_q = [nrns_q]

    nrns_q_dotprop = {}

    pool = Pool( processes=processes )
    neuron_to_dotprop_cond = partial( neuron_to_dotprop, resample_distance=resample_distance, min_strahler=min_strahler, num_nn=num_nn)
    # if min_strahler is None:
    #     nrns_q_dotprop_list = pool.map( neuron_to_dotprop_cond, nrns_q )
    # else:
    #     nrns_q_dotprop_list = pool.map( neuron_to_dotprop_cond, [nrn.strahler_filter(min_strahler) for nrn in nrns_q] )
    nrns_q_dotprop_list = pool.map( neuron_to_dotprop_cond, nrns_q )
    nrns_q_dotprop = {nrn.id:nrns_q_dotprop_list[ii] for ii, nrn in enumerate(nrns_q) }

    if nrns_t is None:
        nrns_t = nrns_q
        nrns_t_dotprop = nrns_q_dotprop
    else:
        nrns_t_dotprop = {}
        for nrn in nrns_t:
            # if min_strahler is None:
            #     nrns_t_dotprop_list = pool.map( neuron_to_dotprop_cond, nrns_t )
            # else:
            #     nrns_t_dotprop_list = pool.map( neuron_to_dotprop_cond, [nrn.strahler_filter(min_strahler) for nrn in nrns_t] )
            nrns_t_dotprop_list = pool.map( neuron_to_dotprop_cond, nrns_t )
            nrns_t_dotprop = {nrn.id:nrns_t_dotprop_list[ii] for ii, nrn in enumerate(nrns_t) }

    queries = []
    targets = []
    similarities = []
    args = []

    for nrn_q in nrns_q:
        q_name = name_number( nrn_q )
        for nrn_t in nrns_t:
            t_name = name_number( nrn_t )
            queries.append(q_name)
            targets.append(t_name)
            args.append((nrns_q_dotprop[nrn_q.id], nrns_t_dotprop[nrn_t.id], score_lookup, resample_distance, num_nn, normalize) )

    similarities = pool.map( _nblast_neuron_pair_for_mp, args  )
    pool.close()
    df = pd.DataFrame({'Queries':queries, 'Targets':targets, 'S':similarities}).reindex(columns=['Queries','Targets','S'])
    return df


def exact_nblast(score_lookup, nrns_q, nrns_t, resample_distance=1000, num_nn=5, min_strahler=None, processes=4 ):
    """
        Compute a symmetric normalized version of NBLAST to test for "exact
        matches", as the geometric mean of query to target and target to query.
        NBLAST values below 0 (even more dissimilar than random) are filled to 0.
    """
    Sqt = nblast_neurons(
                        score_lookup,
                        nrns_q,
                        nrns_t,
                        resample_distance=resample_distance,
                        num_nn=num_nn,
                        min_strahler=min_strahler,
                        processes=processes,
                        normalize=True
                        ).pivot(  index='Queries',
                                  columns='Targets',
                                  values='S')
    Stq = nblast_neurons(
                        score_lookup,
                        nrns_t,
                        nrns_q,
                        resample_distance=resample_distance,
                        num_nn=num_nn,
                        min_strahler=min_strahler,
                        processes=processes,
                        normalize=True
                        ).pivot(  index='Targets',
                                  columns='Queries',
                                  values='S' )
    return Sqt.multiply( Stq ).apply( np.sqrt ).fillna( 0 )

def match_report( nrns_q, Sb, min_similarity = 0.4 ):
    """
        Takes a datatable of normalized scores structured like the output of
        exact_nblast and gives the best match(es) for every query with scores
        above a specified similarity.
    """
    matches = {}
    for nrn in nrns_q:
        nrn_key = name_number( nrn )
        ordered_matches = Sb.T[nrn_key].sort_values(ascending=False)
        best_matches = ordered_matches[ordered_matches>min_similarity]
        if len(best_matches) == 0:
            matches[nrn.id] = "No sufficiently good matches for " + nrn_key
        else:
            output = best_matches.to_frame()
            matches[nrn.id] =  output
    return matches

def name_number( nrn ):
    return nrn.name + ' (' + str(nrn.id) + ')'

# def name_number_to_id( n ):
#     re.
#     return 

def similarity_matrix_to_adjacency( Sb, min_similarity=0.4 ):
    """
        Bring a similarity matrix into a networkx graph format (e.g. for matching)
    """    
    B = nx.Graph()

    nodes_from = list(Sb.index.values)
    nodes_to = list(Sb.columns.values)

    B.add_nodes_from( nodes_from, bipartite=0 )
    B.add_nodes_from( nodes_to, bipartite=1 )

    Sb_edgelist = Sb.unstack().swaplevel('Targets','Queries')
    for node1 in nodes_from:
        for node2 in nodes_to:
            if Sb_edgelist[node1][node2] > min_similarity:
                B.add_edge(node1,node2,weight = Sb_edgelist[node1][node2] )

    remove_isolates = list( nx.isolates(B) )            
    B.remove_nodes_from( remove_isolates )
    return B

def max_match_similarity(Sb, min_similarity=0.4, enforce_match=None, enforce_match_val=1 ):

    tempSb = copy.deepcopy(Sb)
    if enforce_match is not None:
        for force_match in enforce_match:
            tempSb[force_match[1]][force_match[0]] = enforce_match_val

    B = similarity_matrix_to_adjacency( tempSb, min_similarity=min_similarity)
    Qset = [n for n,d in B.nodes(data=True) if d['bipartite']==0]

    matching = nx.algorithms.matching.max_weight_matching( B )
    name_q = []
    name_t = []
    match_val = []
    for match in matching:
        if match[0] in Qset:
            name_q.append(match[0])
            name_t.append(match[1])
            match_val.append(Sb[match[1]][match[0]])
        else:
            name_q.append(match[1])
            name_t.append(match[0])
            match_val.append(Sb[match[0]][match[1]])

    return pd.DataFrame({'Query':name_q,'Target':name_t,'Max S':match_val}).sort_values(by='Max S',ascending=False)
