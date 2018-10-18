import catalysis as cat
import catalysis.pynblast as pynblast

import numpy as np
import scipy as sp
import pandas as pd

from itertools import chain, cycle
from sklearn import cluster
import sys
epsilon = sys.float_info.epsilon


def import_landmarks( CatmaidInterface, filename ):
	'''
	Import a landmark csv file with bundle entry points.

	Parameters
	----------
	filename : string
		File name of csv file

	Returns
	-------
	hemilateral_groups : dict of dicts

	'''
	hemilateral_groups = CatmaidInterface.match_groups_from_select_annotations( wb_match, lineage_parser )



	with open(filename) as fid:
	    lin_landmarks = pd.read_csv(fid)

	no_landmark_lins = []
	for lin in hemilateral_groups:
	    slc = lin_landmarks.lineage_name.str.find(lin)==0
	    if not any( slc ):
	        no_landmark_lins.append(lin)
	    else:
	        hemilateral_groups[lin]['l_loc'] = np.array(lin_landmarks[slc][['x_left', 'y_left', 'z_left']])
	        hemilateral_groups[lin]['r_loc'] = np.array(lin_landmarks[slc][['x_right', 'y_right', 'z_right']])
	return hemilateral_groups, no_landmark_lins