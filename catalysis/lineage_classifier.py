import pandas as pd
import scipy as sp
import numpy as np
import catalysis as cat
import catalysis.transform as transform
import catalysis.pynblast as pynblast
from itertools import chain
import sys
epsilon = sys.float_info.epsilon


def lineage_table( hemilateral_groups, lin_landmarks, side_name ):
	"""
		Build a table of lineage entry points
	"""
	lineages = sorted(list(hemilateral_groups.keys()))
	        
	no_landmark_lins = []
	for lin in hemilateral_groups:
	    slc = lin_landmarks.lineage_name.str.find(lin)==0
	    if not any( slc ):
	        no_landmark_lins.append(lin)
	    else:
	        hemilateral_groups[lin]['l_loc'] = np.array(lin_landmarks[slc][['x_left', 'y_left', 'z_left']])
	        hemilateral_groups[lin]['r_loc'] = np.array(lin_landmarks[slc][['x_right', 'y_right', 'z_right']])
	        
	all_locations = []
	lin_list = []
	side_list = []
	for side in side_name:
	    for lin in hemilateral_groups.keys():
	        if side in hemilateral_groups[lin].keys():
	            all_locations.append( hemilateral_groups[lin][side][0] )
	            lin_list.append( lin )
	            side_list.append( side[0] )

	lin_df = pd.DataFrame({'lin':lin_list,'side':side_list,'xyz':all_locations})
	return lin_df

def compute_initial_segment_dotprops( nrns, l_span, resample_distance):
	"""
		From a list of neurons, compute the dotprops associated with an initial span of the arbor connected to the soma
		Parameters
		----------
		nrns : NeuronList
			Neurons for which to compute

		l_span : number
			Length of the initial segment in nm

		resample_distance : number
			How frequently to resample the dotprop, in nm.

		Returns
		-------
		dict
			Dictionary of id:dotprop for the neurons provided.		
	"""
	initial_seg_dotprops = {}
	for nrn in nrns:
	    d_close = nrn.dist_to_root() < l_span
	    initial_nodes = [nid for nid in nrn.minimal_paths()[0] if d_close[ nrn.node2ind[nid]]]
	    xyz_init = [nrn.nodeloc[nid] for nid in initial_nodes]
	    dp_init = pynblast.dotprop_path( xyz_init, resample_distance=resample_distance)
	    initial_seg_dotprops[nrn.id] = dp_init
	return initial_seg_dotprops

def compute_conditional_distributions( nblast_mat, lin_dict, lin_name2id, rel_lins, skip_skids=None):
	"""
		From a list of within and surround ids, find the distribution of median NBLAST similarities.

		Parameters
		----------
		nblast_mat : DataFrame
			DataFrame containing the nblast scores between all skids nearby.

		lin_dict : dict
			Dictionary of annotation id to skeleton ids within that annotation.

		lin_name2id : dict
			Dictionary mapping lineage name to annotation id

		rel_lins : list
			list of lineage names that are potential candidates
		
		Returns
		-------

		P_match
			Dict indexed by lineage name, values give median nblast scores from skeletons in the
			same lineage to other skeletons in that lineage.

		p_unmatch
			Dict indexed by lineage name, values give median nblast scores from skeletons in other lineages to index group.
	"""	
	P_match = {}
	for lin in rel_lins:
		within_ids = list(set(lin_dict[lin_name2id[lin]]).difference([skip_skids]))
		P_match[lin] = []
		for skid1 in within_ids:
			all_sim = [nblast_mat[skid1][skid2] for skid2 in list(set(within_ids).difference(set([skid1]))) ]
			P_match[lin].append( np.median(all_sim) )

	all_ids = set([skid for skid in chain.from_iterable(lin_dict.values())])

	P_unmatch = {}
	for lin in rel_lins:
		within_ids = lin_dict[lin_name2id[lin]]
		surround_ids =  all_ids.difference( lin_dict[lin_name2id[lin]] )
		P_unmatch[lin] = []
		for skid1 in surround_ids:
			all_sim = [ nblast_mat[skid1][skid2] for skid2 in within_ids ]
			P_unmatch[lin].append( np.median(all_sim) )

	return P_match, P_unmatch

def fit_gaussian_kdes( P_dat ):
	kdes_dat = []
	kde_dat_norm = []

	for ind, lin in enumerate(P_dat):
		kdes_dat.append( sp.stats.gaussian_kde( np.array(P_dat[lin]) ) )
		kde_dat_norm.append( kdes_dat[ind].integrate_box_1d(0,1) )
	return kdes_dat, kde_dat_norm

def lineage_likelihood_ratios(rel_skid, nblast_mat, lin_dict, lin_name2id, rel_lins, kdes_match, kdes_match_norm, kdes_unmatch, kdes_unmatch_norm):
	lhds = []
	for ind, lin in enumerate(rel_lins):
	    d_toA = np.median(np.array([nblast_mat[nblast_mat.index==rel_skid][skid].values for skid in lin_dict[lin_name2id[lin]]]) )
	    num =  kdes_match[ind].evaluate( np.round(d_toA,2) )/kdes_match_norm[ind]
	    denom =  kdes_unmatch[ind].evaluate( np.round(d_toA,2) )/kdes_unmatch_norm[ind]
	    lhds.append( num / denom )
	lhds = np.log(np.array(lhds).reshape(-1))
	lhds_ratio = np.round( lhds / np.max(lhds), 3)
	base_skid = len(rel_lins) * [rel_skid]
	return pd.DataFrame( {'Lineage':rel_lins, 'LikelihoodRatio':lhds, 'FractionOfBest':lhds_ratio, 'SkeletonId':base_skid})
