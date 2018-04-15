import pandas as pd
import scipy as sp
import numpy as np
import networkx as nx
import catalysis as cat
import re
import copy
from bisect import bisect
from multiprocessing import Pool
from functools import partial
import catalysis.transform as transform

def nblast_neuron_pair( nrn_q,
                        nrn_t,
                        score_lookup,
                        resample_distance = 1000,
                        num_nn = 5,
                        bidirectional = False,
                        normalize = False ):
    """
        Find the NBLAST score for two neurons, given a score matrix object
    """
    d_and_udotv = neuron_comparison_nblast_components( nrn_q, nrn_t, resample_distance=resample_distance, num_nn = num_nn )
    S_a = nblast_dist_fun( d_and_udotv, score_lookup )
    if bidirectional:
        d_and_udotv_b = neuron_comparison_nblast_components( nrn_t, nrn_q, resample_distance=resample_distance, num_nn = num_nn )
        S_b = nblast_dist_fun( d_and_udotv, score_lookup )
        S_a = S_a / max_blast_score( len(d_and_udotv), score_lookup )
        S_b = S_b / max_blast_score( len(d_and_udotv), score_lookup )
        S_a = max(S_a, 0)
        S_b = max(S_b, 0)
        S = np.sqrt( S_a * S_b )
    elif normalize:
        S = S_a / max_blast_score( len(d_and_udotv), score_lookup )
    else:
        S = S_a
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

    # Check if neurons are effectively emtpy, likely due to Strahler pruning.
    if len( nrn_q_dotprop ) > 1 and len(nrn_t_dotprop) > 1:
        d_and_udotv = neuron_comparison_nblast_components( nrn_q_dotprop, nrn_t_dotprop, resample_distance=resample_distance, num_nn = num_nn, as_dotprop = True )
        S = nblast_dist_fun( d_and_udotv, score_lookup )
        
        if normalize:
            S = S / max_blast_score( len(d_and_udotv), score_lookup )
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

def nblast_dist_fun_local( d_and_udotv, score_lookup ):
    Sloc = []
    if np.shape(d_and_udotv)[1] != 2:
        raise ValueError( "Scores must only have two components")
    for row in d_and_udotv:
        Sloc.append( score_lookup.score( d=row[0], udotv=row[1] ) )
    return np.array( Sloc )

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
    nrns_q_dotprop_list = pool.map( neuron_to_dotprop_cond, nrns_q )
    nrns_q_dotprop = {nrn.id:nrns_q_dotprop_list[ii] for ii, nrn in enumerate(nrns_q) }

    if nrns_t is None:
        nrns_t = nrns_q
        nrns_t_dotprop = nrns_q_dotprop
    else:
        nrns_t_dotprop = {}
        for nrn in nrns_t:
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

def soma_distance( nrns_q, nrns_t=None ):

    if type(nrns_q) is 'NeuronObj':
        nrns_q = [nrns_q]

    if nrns_t is None:
        nrns_t = nrns_q

    queries = []
    targets = []
    soma_distances = []
    for nrn_q in nrns_q:
        q_name = name_number( nrn_q )
        xyz_q = nrn_q.soma_location()
        for nrn_t in nrns_t:
            t_name = name_number( nrn_t )
            xyz_t = nrn_t.soma_location()
            queries.append(q_name)
            targets.append(t_name)
            soma_distances.append(np.linalg.norm(xyz_q-xyz_t))
    df = pd.DataFrame({'Queries':queries, 'Targets':targets, 'soma_distance':soma_distances}).reindex(columns=['Queries','Targets','soma_distance'])
    return df

def exact_nblast(score_lookup,
                nrns_q,
                nrns_t,
                resample_distance=1000,
                num_nn=5,
                min_strahler=None,
                min_length=None,
                processes=4 ):
    """
        Compute a symmetric normalized version of NBLAST to test for "exact
        matches", as the geometric mean of query to target and target to query.
        NBLAST values below 0 (even more dissimilar than random) are filled to 0.
    """

    if min_length is not None:
        nrns_q = cat.filter_neurons_by_length( nrns_q, min_length )
        nrns_t = cat.filter_neurons_by_length( nrns_t, min_length )

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
    Stq[Stq<0]=0
    Sqt[Sqt<0]=0

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

def name_number_to_id( name_num ):
    id_match = re.search('\((\d*)\)$',name_num)
    return int(id_match.groups()[0])

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
    match_ids_q = []
    match_ids_t = []
    for match in matching:
        if match[0] in Qset:
            name_q.append(match[0])
            name_t.append(match[1])
            match_val.append(Sb[match[1]][match[0]])
            match_ids_q.append( name_number_to_id(match[0]) )
            match_ids_t.append( name_number_to_id(match[1]) )
        else:
            name_q.append(match[1])
            name_t.append(match[0])
            match_val.append(Sb[match[0]][match[1]])
            match_ids_q.append( name_number_to_id(match[1]) )
            match_ids_t.append( name_number_to_id(match[0]) )
    match_df = pd.DataFrame({'Query_name':name_q,'Query_id':match_ids_q,'Target_name':name_t,'Target_id':match_ids_t,'S':match_val}).sort_values(by='S',ascending=False).reset_index(drop=True)
    return match_df.reindex(['S',
                      'Query_name',
                      'Query_id',
                      'Target_name',
                      'Target_id'], axis = 1)

def compare_partners( score_lookup,
                      nrn_q,
                      nrn_t,
                      CatmaidInterface,
                      connection_type,
                      from_group,
                      to_group,
                      resample_distance=1000,
                      num_nn=5,
                      min_strahler=None,
                      min_length=None,
                      ntop_q=2,
                      kmin_f=0.5,
                      kmin_t=3,
                      normalized=False,
                      return_full_similarity=False,
                      return_neurons=False,
                      contralateral=True  ):
    """
        Finds a maximum max of NBLAST scores for partners of a queried neuron.
        Parameters
        ----------
        score_lookup : ScoreMatrixLookup
            nblast score matrix

        nrn_q : NeuronObj
            Neuron to query

        nrn_t : NeuronObj
            Target neuron

        CatmaidInstance : CatmaidDataInterface
            Needed to query the server for new data

        connection_type : 'presynaptic' or 'postsynaptic'
            Defines direction of connections to look at.

        ntop_q : Positive Int
            Number of top connections to consider

        kmin_f : float (between 0 and 1)
            Fraction of query synapses to include in target search. E.g. If the
            connection is X synapses in the minimum weight connection in the
            query, look for partners with more than kmin_f * X synapses in
            target.

    """

    # Get top partners of query neuron
    partner_ids_all_q = nrn_q.synaptic_partners( connection_type=connection_type )
    ntop_q = min(ntop_q,len(partner_ids_all_q))
    partner_nrns_q = cat.NeuronList.from_id_list( id_list = partner_ids_all_q[0:ntop_q,0],
                                                  CatmaidInterface=CatmaidInterface,
                                                  with_tags=True,
                                                  with_annotations=False )

    min_synapses = int(max( kmin_t, kmin_f*partner_ids_all_q[ntop_q-1,1] ) )

    # Get (larger list of) top partners of target neuron
    partner_ids_all_t = nrn_t.synaptic_partners( connection_type=connection_type,
                                                 min_synapses=min_synapses,
                                                 normalized=normalized )
    partner_nrns_t = cat.NeuronList.from_id_list( id_list = partner_ids_all_t[:,0],
                                                  CatmaidInterface=CatmaidInterface,
                                                  with_tags=True,
                                                  with_annotations=False )
    partner_nrns_t_transformed = transform.transform_neuronlist(
                                                partner_nrns_t,
                                                from_group=from_group,
                                                to_group=to_group,
                                                CatmaidInterface=CatmaidInterface,
                                                contralateral=contralateral)


    Sb = exact_nblast( score_lookup,
                       partner_nrns_q,
                       partner_nrns_t_transformed,
                       resample_distance=resample_distance,
                       num_nn=num_nn,
                       min_strahler=min_strahler,
                       min_length=min_length )
    matching = max_match_similarity( Sb )

    # Add synapses 
    q_syn = []
    for nid in matching['Query_id']:
        q_syn.append(partner_ids_all_q[partner_ids_all_q[:,0]==nid,1][0])
    t_syn = []
    for nid in matching['Target_id']:
        t_syn.append(partner_ids_all_t[partner_ids_all_t[:,0]==nid,1][0])
    syn_df = pd.DataFrame({'Query_synapses': q_syn, 'Target_synapses': t_syn})
    matching = matching.join(syn_df)
    matching = matching.reindex(['S',
                      'Query_name',
                      'Query_id',
                      'Target_name',
                      'Target_id',
                      'Query_synapses',
                      'Target_synapses'], axis = 1)
    
    out = {}
    out['matching'] = matching
    if return_full_similarity:
        out['similarity'] = Sb
    
    if return_neurons:
        out['partners_q'] = partner_nrns_q.slice_by_id(matching['Query_id'])
        out['partners_t'] = partner_nrns_t_transformed.slice_by_id(matching['Target_id'])

    return out