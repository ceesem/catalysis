import catalysis.catmaid_interface as ci
import catalysis.neurons as na
import catalysis.pynblast as pynblast

from collections import defaultdict
import pandas as pd
import numpy as np
from numpy import NaN

def root_is_soma( nrn ):
    """
    Check that a neuron's root is tagged 'soma' and no other node is tagged
    similarly.

    Parameters
    ----------
    nrn : Neuron object.
        Neuron to test

    Returns
    ----------
    Boolean
        True if root is soma, false otherwise.
    """
    val = False
    if 'soma' in nrn.tags.keys():
        if len(nrn.tags['soma']) == 1:
            if nrn.tags['soma'][0] == nrn.root:
                val = True

    if 'to nerve' in nrn.tags.keys():
        if len(nrn.tags['to nerve']) == 1:
            if nrn.tags['to nerve'][0] == nrn.root:
                val = True

    return val

def open_ends( nrn, include_uncertain = True ):
    """
        Check that end nodes fall into the appropriate 'blessed' tags.
        Parameters
        ----------
        nrn : Neuron object
            Neuron to test

        include_uncertain : Boolean (Default : True)
            Boolean variable associated with whether to include "uncertain ends"
            and "uncertain continuation" in the statistics.

        Returns
        ----------
        double
            Fraction of open ends that are tagged complete.

        int
            Number of open ends.

        int
            Number of total ends.

    """
    if include_uncertain:
        blessed_tags = ['ends',
                        'soma',
                        'not a branch',
                        'uncertain end',
                        'uncertain continuation']
    else:
        blessed_tags = ['ends',
                        'soma',
                        'not a branch']

    open_end_nodes = set( nrn.find_end_nodes() )
    num_ends = len(open_end_nodes)

    for tag in blessed_tags:
        if tag in nrn.tags.keys():
            open_end_nodes.difference_update( set(nrn.tags[tag]) )

    rem_ends = len(open_end_nodes)
    frac_open = (rem_ends+0.0) / (num_ends + 0.0)
    return frac_open, rem_ends, num_ends

def fragment_test( nrn ):
    """
        A fragment is defined as having no soma
    """
    if 'soma' in nrn.tags.keys() or 'to nerve' in nrn.tags.keys():
        return False
    else:
        return True

def property_summary_exact( nrns ):
    """
    Generate a readable table summarizing the completeness of reconstruction for
    a group of neurons.
        Parameters
        ----------
        nrns : NeuronDictObject
                Collection of neurons to analyze

        Returns
        ----------
        DataFrame
            Data frame describing completeness of neurons
    """

    ids = [nrn.id for nrn in nrns]
    names = [nrn.name for nrn in nrns]
    fraction_open_ends = [ open_ends(nrn)[0] for nrn in nrns]
    num_open_ends = [ open_ends(nrn)[1] for nrn in nrns]
    num_ends = [open_ends(nrn)[2] for nrn in nrns]
    correct_root = [ root_is_soma(nrn) for nrn in nrns ]
    node_count = [ nrn.node_count() for nrn in nrns ]
    is_fragment = [ fragment_test(nrn) for nrn in nrns ]
    dat = { 'names':names,
            'fraction_open_ends':fraction_open_ends,
            'num_open_ends':num_open_ends,
            'total_ends':num_ends,
            'rooted_at_soma':correct_root,
            'node_count':node_count,
            'is_fragment':is_fragment}

    if nrns.CatmaidInterface is not None:
        review_status = nrns.CatmaidInterface.get_review_status(ids)
        rev_frac = [ (review_status[str(id)][1]+0.0)
                     / (review_status[str(id)][0]+0.0) for id in ids ]
        dat['review_fraction'] = rev_frac
    return pd.DataFrame(dat, index=ids)

def property_summary_estimated( ids, CatmaidInterface ):
    """
        A thinner version of property_summary_exact that only retrieves specific
        relevent info, but only approximates total number of end nodes and is
        thus faster than completeness_summary_exact.

        Parameters
        ----------
        ids : list of ids
            Ids for neurons to query completeness for.

        CatmaidInterface : CatmaidDataInterface
            CatmaidDataInterface for a specific project.

        Returns
        -------
        DataFrame
            DataFrame describing the completeness status of neurons

    """
    namedict = CatmaidInterface.get_neuron_names( ids )
    names = {int(id) : namedict[id] for id in namedict}
    fraction_open_ends = []
    num_open_ends = []
    num_ends = []
    correct_root = []
    node_count = []
    is_fragment = []

    for id in ids:
        noe = len( CatmaidInterface.get_open_ends( id ) )
        nce = len( CatmaidInterface.get_closed_ends( id ) )
        sn = CatmaidInterface.tag_query_for_skeleton( id, '^soma$|^out to nerve$')
        rn = CatmaidInterface.root_for_skeleton( id )['root_id']
        fraction_open_ends.append( (noe+0.0)/(noe+nce+0.0) )
        num_open_ends.append( noe )
        num_ends.append( noe+nce )
        if rn in sn:
            correct_root.append( True )
        else:
            correct_root.append( False )
        node_count.append( CatmaidInterface.node_count( id )['count'] )
        if len(sn) == 0:
            is_fragment.append( True )
        else:
            is_fragment.append( False )

    dat = { 'names':names,
            'fraction_open_ends':fraction_open_ends,
            'num_open_ends':num_open_ends,
            'total_ends':num_ends,
            'rooted_at_soma':correct_root,
            'node_count':node_count,
            'is_fragment':is_fragment}

    review_status = CatmaidInterface.get_review_status(ids)
    rev_frac = [ (review_status[str(id)][1]+0.0)
                  / (review_status[str(id)][0]+0.0) for id in ids ]
    dat['review_fraction'] = rev_frac

    return pd.DataFrame(dat, index=ids)

def completion_categories( property_summary = None,
                           completeness_threshold = 0.97,
                           review_threshold = 0.97 ):
    """
        Returns a dict describing which neurons belong to which completeness
        categories. This takes a property summary as its input.

        Parameters
        ----------
        property_summary : DataFrame
            A property summary from either property_summary_exact or
            property_summary_estimated.

        completeness_threshold : float (optional, default = 0.97)
            The mininum fraction of complete end nodes a neuron must have to be
            considered complete.

        review_threshold : float (optional, default = 0.97)
            The minimum fraction of reviewed nodes a neuron must have to be
            considered reviewed.

        Returns
        -------
        dict
            Dict of categories (keys), with values being a list of ids.
            Categories are:
                Untraced : Only a single untagged node.
                Incomplete Fragment : Fragment (no soma) with open ends (presumed unfinished)
                Complete Fragment : Fragment with no more open ends (presumed attempted and failed)
                Incomplete Neuron : Neuron (unique soma), but open ends.
                Complete Neuron : Neuron with no open ends.
                Reviewed Complete Neuron : Neuron with no open ends and substantially reviewed. (if include_reviewed = True)
    """

    categories = {'Untraced':[],
                  'Incomplete fragment':[],
                  'Complete fragment':[],
                  'Incomplete neuron':[],
                  'Complete neuron':[],
                  'Reviewed complete neuron':[]
                  }

    if property_summary is not None:
        for id in property_summary.index.values:
            if property_summary.loc[id]['node_count'] == 1:
                categories['Untraced'].append(id)
            elif property_summary.loc[id]['is_fragment']:
                if property_summary.loc[id]['fraction_open_ends'] <= 1-completeness_threshold:
                    categories['Complete fragment'].append(id)
                else:
                    categories['Incomplete fragment'].append(id)
            elif property_summary.loc[id]['fraction_open_ends'] > 1-completeness_threshold:
                categories['Incomplete neuron'].append(id)
            elif property_summary.loc[id]['review_fraction'] >= review_threshold:
                categories['Reviewed complete neuron'].append(id)
            else:
                categories['Complete neuron'].append(id)
    return categories


def category_summary( categories, syn_df=None, as_df = True, nans=False ):
    """
    For a given categorization and synaptic connectivity table, summarize
    cateogries by number and synaptic count.

    Parameters
    ----------
    categories : dict
        Dict keyed by category name with values being lists of neuron ids.

    syn_df : dataframe (optional, default is None)
        DataFrame representing a table of synaptic connectivity formated like in
        synaptic_partner_tables.

    as_df : Boolean (optional, default is True)
        Determines if the response is a dataframe or remains a dict.

    nans : Boolean (optional, default is False)
        Returns only nans. Useful to generate reports that have the right shape,
        but no data.

    Returns
    ----------
    DataFrame (or dict, if as_df=False)
        DataFrame indexed by categories, with columns being number of neurons
        and synapses.

    """

    ids_cat = set( [item for sublist in [categories[cat] for cat in categories]
                    for item in sublist if item is not None] )
    if syn_df is not None:
        # Remove any None objects that could happen in case of
        # connector with no other annotation
        ids_syn = set( filter( None.__ne__, syn_df.index.values ) ) 
        if ids_cat != ids_syn:
            print(ids_cat)
            print(ids_syn)
            raise ValueError("IDs in categories and dataframe must be the same")

    cat_by_syn = {}
    cat_by_num = {}
    for cat in categories:
        if nans:
            cat_by_num[cat] = NaN
            cat_by_syn[cat] = NaN
        elif len( categories[cat] ) > 0:
            if syn_df is not None:
                cat_by_syn[cat] = syn_df.loc[ categories[cat] ].sum().sum()
            else:
                cat_by_syn[cat] = NaN
            cat_by_num[cat] = len( categories[cat] )
        else:
            if syn_df is not None:
                cat_by_syn[cat] = 0
            else:
                cat_by_syn[cat] = NaN
            cat_by_num[cat] = 0

    if as_df:
        return pd.DataFrame(
                    {'Synapses': cat_by_syn,'Number':cat_by_num} ).reindex(
                        ['Untraced',
                         'Incomplete fragment',
                         'Incomplete neuron',
                         'Complete fragment',
                         'Complete neuron',
                         'Reviewed complete neuron'] )
    else:
        return cat_by_syn, cat_by_num

def category_summary_from_neurons( nrns, estimate_partner_completion = True, include_presynaptic = False, include_postsynaptic = False, max_neurons_per_post = 50 ):
    """
        Generate category reports about neurons and their partners from a list of neurons.

        Parameters
        ----------
            nrns : NeuronList
                NeuronList forming the basis for the category summary.

            estimate_partner_completion : Boolean (default is True)
                If properties of partners are computed, this value determines if the estimated or the exact property function is used.

            include_presynaptic : Boolean (default is False)
                Determines if an estimate of the properties of the presynaptic neurons is included.

            include_postsynaptic : Boolean (default is False)
                Determines if an estimate of the properties of the postsynaptic neurons is included.

            as_df : Boolean (defualt is True)
                Determines if the result is returned as a DataFrame or a dict.

        Returns
        -------
            dict
                Dict with keys 'Base', 'Inputs', and 'Outputs', each containing the results of category_summary on the main list of neurons,
                the collection of presynaptic neurons (Inputs), or the collection of postsynaptic neurons (Outputs)
    """

    main_cats = completion_categories( property_summary_exact( nrns ) )
    main_report = category_summary(main_cats)

    input_df, output_df = na.synaptic_partner_tables( nrns, include_presynaptic=include_presynaptic, include_postsynaptic=include_postsynaptic )
    if include_presynaptic:
        print( '     Computing presynaptic categories...')
        pre_ids = [id for id in input_df.index.values if id is not None]
        if estimate_partner_completion:
            input_neurons = na.NeuronList.from_id_list( pre_ids, nrns.CatmaidInterface, max_neurons_per_post=max_neurons_per_post )
            input_cats = completion_categories( property_summary_exact( input_neurons ) )
        else:
            input_cats = completion_categories( property_summary_estimated( pre_ids, nrns.CatmaidInterface ) )
        input_report = category_summary(input_cats, input_df)

    else:
        input_report = category_summary( main_cats, nans = True )

    if include_postsynaptic:
        print( '     Computing postsynaptic categories...')
        post_ids = [id for id in output_df.index.values if id is not None]
        if estimate_partner_completion:
            output_neurons = na.NeuronList.from_id_list( post_ids, nrns.CatmaidInterface, max_neurons_per_post=max_neurons_per_post )
            output_cats = completion_categories( property_summary_exact( output_neurons ) )
        else:
            output_cats = completion_categories( property_summary_estimated( post_ids, nrns.CatmaidInterface ) )
        output_report = category_summary(output_cats, output_df)
    else:
        output_report = category_summary( main_cats, nans = True )

    report = {'Base': main_report, 'Inputs': input_report, 'Outputs': output_report }
    return report

def completeness_report( CatmaidInterface,
                   annos = [],
                   id_list = [],
                   estimate_partner_completion = True,
                   include_presynaptic = False,
                   include_postsynaptic = False,
                   complete_categories = None,
                   max_neurons_per_post = 50):
    """
        Build a report on a set of neurons from annotations or ids, pulling them from a Catmaid instance.

        Parameters
        ----------
            CatmaidInterface : CatmaidDataInterface
                Interface for the catmaid instance to use.

            annos : list of strings or ints (optional, default is [])
                List of annotation ids or names to query.

            id_list : list of ints (optional, default is [])
                List of Neuron ids to query.

            estimate_partner_completion : Boolean (default = True)
                Boolean value selecting whether partners have estimated or exact completeness.

            include_presynaptic : Boolean (default = False)
                Determines whether the set of presynaptic neurons are queried or not.

            include_postsynaptic : Boolean (default = False)
                Determines whether the set of postsynaptic neurons are queried or not.

            complete_categories : List of strings (optional, default = None)
                Overrides the default categories that are considered 'complete'.

            Returns
            -------
            DataFrame
                DataFrame summarizing the completion status of neurons and, if desired, the set of partners.
    """
    if type(annos) is str:
        annos = [annos]

    if len(id_list) > 0 and len(annos) > 0:
        id_tag = ' and Specified Ids'
    elif len(id_list) > 0 and len(annos)==0:
        id_tag = ', '.join( id_list )
    else:
        id_tag = ''
    if len(annos) > 1:
        anno_tag = ' and '.join(annos)
    elif len(annos) > 0:
        anno_tag = str( annos[0] )
    else:
        anno_tag = {}
    anno_name = anno_tag + id_tag

    id_list = list( set( id_list + CatmaidInterface.get_ids_from_annotations(annos, flatten=True) ) )
    nrns = na.NeuronList.from_id_list( id_list, CatmaidInterface, max_neurons_per_post=max_neurons_per_post )
    cats = category_summary_from_neurons( nrns, include_presynaptic = include_presynaptic, include_postsynaptic = include_postsynaptic )

    return report_from_summary( cats, anno_name=anno_name )

def report_from_summary( category_summary, complete_categories = None, anno_name=None):
    """
        Given a category summary, generate a report DataFrame.

        Parameters
        ----------
        category_summary : Dict
            Dict coming out of category_summary_from_neurons

        complete_categories : List of str
            Overriding list of categories to be considered complete.

        anno_name : string
            Name describing the group, for cosmetic use in the dataframe.

        Returns
        -------
        DataFrame
            DataFrame summarizing completion status.
    """
    if complete_categories is None:
        complete_categories = ['Complete fragment', 'Complete neuron', 'Reviewed complete neuron']
    if anno_name is None:
        anno_name = 'Group'

    base_complete, base_inc, base_frac, base_complete_syn, base_frac_syn = _report_categories( category_summary['Base'], complete_categories=complete_categories )
    input_complete, input_inc, input_frac, input_complete_syn, input_frac_syn = _report_categories( category_summary['Inputs'], complete_categories=complete_categories )
    output_complete, output_inc, output_frac, output_complete_syn, outputfrac_syn = _report_categories( category_summary['Outputs'], complete_categories=complete_categories )

    num_master = {anno_name:base_complete, 'Inputs':input_complete, 'Outputs':output_complete}

    inc_master = {anno_name:base_inc,
                  'Inputs':input_inc,
                  'Outputs':output_inc }

    frac_master = {anno_name: base_frac ,
                    'Inputs': input_frac,
                    'Outputs': output_frac}

    num_master_syn = {anno_name:base_complete_syn, 'Inputs':input_complete_syn, 'Outputs':output_complete_syn}

    frac_master_syn = {anno_name:base_complete_syn / sum(category_summary['Base'].Synapses),
                    'Inputs': input_complete_syn / sum(category_summary['Inputs'].Synapses),
                    'Outputs': output_complete_syn / sum(category_summary['Outputs'].Synapses)}
    return pd.DataFrame({'Number Complete':num_master, 'Number Incomplete': inc_master, 'Fraction Complete':frac_master,
                         'Synapses Complete':num_master_syn, 'Fraction Synapses Complete':frac_master_syn})


def _report_categories( dat, complete_categories ):
    """
        Compute needed details of categories to generate report.
    """
    comp_num = sum( [ dat.Number[cat] for cat in complete_categories] )
    incomp_num = sum( dat.Number ) - comp_num
    frac_num = comp_num / sum(dat.Number)
    comp_syn = sum( [ dat.Synapses[cat] for cat in complete_categories] )
    frac_syn = comp_syn / sum(dat.Synapses)
    return comp_num, incomp_num, frac_num, comp_syn, frac_syn

def match_groups( id_list1, id_list2, match_via, CatmaidInterface, anno_reference='names' ):
    """
        Given two lists of neurons, match their elements if they share an annotation (such as cell type) indicated by a specific metaannotation.

        Parameters
        ----------
            id_list1 : list of ints
                List of skeleton ids in the first group

            id_list2 : list of ints
                List of skeleton ids in the second group

            match_via : string or int
                Annotation (as name or id) that annotates the annotatoins to match.

            CatmaidInterface : CatmaidDataInterface
                Interface for the Catmaid instance to query.

            anno_reference : 'names' or 'ids' (optional, default is 'names')

        Returns
        -------
            dict
                Match report, indexed by annotation name or id.
    """
    annos_with_meta = set(CatmaidInterface.get_annotations_from_meta_annotations( match_via, flatten=True ) )
    annos1 = { id: set(CatmaidInterface.get_annotations_for_objects( [id] )).intersection(annos_with_meta) for id in id_list1}
    annos2 = { id: set(CatmaidInterface.get_annotations_for_objects( [id] )).intersection(annos_with_meta) for id in id_list2}

    matches = {}
    for id1 in annos1:
        for id2 in annos2:
            for anno_id in annos1[id1].intersection(annos2[id2]):
                if anno_reference is 'ids':
                    matches[anno_id] = [id1, id2]
                elif anno_reference is 'names':
                    matches[ CatmaidInterface.parse_annotation_list(anno_id, output='names')[0] ] = [id1, id2]

    return matches

def match_report( id_list1, id_list2, match_via, CatmaidInterface, name1='Group 1', name2='Group 2', anno_reference = 'names', skip_annos=None, show_completeness=True ):
    """
            Given two lists of neurons, match their elements if they share an annotation (such as cell type) indicated by a specific metaannotation.

            Parameters
            ----------
                id_list1 : list of ints
                    List of skeleton ids in the first group

                id_list2 : list of ints
                    List of skeleton ids in the second group

                match_via : string or int
                    Annotation (as name or id) that annotates the annotatoins to match.

                CatmaidInterface : CatmaidDataInterface
                    Interface for the Catmaid instance to query.

                name1 = string (Default is 'Group 1')
                    Label for the first group.

                name2 = string (Default is 'Group 2')
                    Label for the second group.

                anno_reference : 'names' or 'ids' (optional, default is 'names')

                skip_annos : list of strings (optional, Default is None)
                    List of annotations of neurons to not include in the final report.

            Returns
            -------
                DataFrame
                    Organized, readable match report

    """
    if skip_annos is not None:
        ids_to_skip = CatmaidInterface.get_ids_from_annotations(skip_annos, flatten=True)
        id_list1 = list( set(id_list1).difference(set(ids_to_skip)))
        id_list2 = list( set(id_list2).difference(set(ids_to_skip)))

    annos = CatmaidInterface.get_annotations()
    rev_dict = { annos[key] : key for key in annos }

    matches = match_groups( id_list1, id_list2, match_via, CatmaidInterface )
    matched = [[],[]]
    match_report1 = {}
    match_report2 = {}
    for anno in matches:
        if type(anno) is int:
            #anno_name = annos[anno_id] + ' (' + str(anno_id) + ')'
            anno_name = rev_dict[anno] + ' (' + str(anno) + ')'
        else:
            anno_name = anno
        matched[0].append( matches[anno][0] )
        matched[1].append( matches[anno][1] )
        match_report1[ anno_name ] = matches[anno][0]
        match_report2[ anno_name ] = matches[anno][1]

    unmatched1 = set(id_list1).difference( set(match_report1.values()) )
    unmatched2 = set(id_list2).difference( set(match_report2.values()) )
    match_report1['Unmatched'] = list(unmatched1)
    match_report2['Unmatched'] = list(unmatched2)

    if show_completeness:
        ps = property_summary_estimated(id_list1+id_list2, CatmaidInterface )
        match_completed1 = _match_completed(ps, match_report1 )
        match_completed2 = _match_completed(ps, match_report2 )

    report = pd.DataFrame( { name1: match_report1, name2: match_report2, name1+'_complete':match_completed1, name2+'_complete':match_completed2} )
    return report

def _match_completed( ps, match_report, min_open_ends=0.05,  ):
    match_completed = dict()
    for lin in match_report:
        if isinstance( match_report[lin], np.int ):
            relids = [match_report[lin]]
        else:
            relids = match_report[lin]
        match_completed[lin] = []
        for skid in relids:
            if ps[ps.index==skid]['is_fragment'].bool() is False and (ps[ps.index==skid]['fraction_open_ends'] < min_open_ends).bool():
                match_completed[lin].append(True)
            else:
                match_completed[lin].append(False)
    return match_completed

def match_report_from_annos( anno1, anno2, match_via, CatmaidInterface, anno_reference = 'names', skip_annos = None, show_completeness=False):
    """
        Given two lists of neurons, match their elements if they share
        an annotation (such as cell type) indicated by a specific
        metaannotation.

        Parameters
        ----------
            anno1 : string
                Name of the annotation to query for group 1.

            anno2 : string
                Name of the annotation to query for group 2.

            match_via : string or int
                Annotation (as name or id) that annotates the annotations to
                match.

            CatmaidInterface : CatmaidDataInterface
                Interface for the Catmaid instance to query.

            anno_reference : 'names' or 'ids' (optional, default is 'names')
                Determines how annotations are refered to, either as strings or
                ids.

        Returns
        -------
            DataFrame
                Organized, human-readable match report

    """
    return match_report( CatmaidInterface.get_ids_from_annotations(anno1,flatten=True),
                         CatmaidInterface.get_ids_from_annotations(anno2,flatten=True),
                         match_via,
                         CatmaidInterface,
                         skip_annos=skip_annos,
                         name1 = anno1,
                         name2 = anno2,
                         show_completeness=show_completeness)

def report_from_annotation_list( anno_list, CatmaidInterface ):
    """
        Generate a completeness report on every annotation in a list of annotations.

        Parameters
        ----------
            anno_list : list of ids or strings
                Annotations to query

            CatmaidInterface : CatmaidDataInterface
                Interaction object for a catmaid instance.

        Returns
        -------
            DataFrame
                DataFrame completion report where each row corresponds to an annotation.
    """
    meta_rep = pd.DataFrame(columns = ['Number Complete','Number Incomplete','Fraction Complete','Synapses Complete','Fraction Synapses Complete'])

    for anno in anno_list:
        rep = completeness_report( CatmaidInterface=CatmaidInterface, annos=[anno] )
        meta_rep = meta_rep.append( rep.iloc[0] )
    return meta_rep[['Fraction Complete','Number Complete','Number Incomplete']]

def report_from_meta( meta, CatmaidInterface):
    """
        Generate a completeness report based on a meta-annotation.

        Parameters
        ----------
            meta : string or id.
                Meta-annotation for which to generate the report.

            CatmaidInterface : CatmaidDataInterface
                Interaction object for a catmaid instance.

        Returns
        -------
            DataFrame
                Report for the annotations within the meta-annotation.
    """
    anno_list = CatmaidInterface.get_annotations_from_meta_annotations(
                    meta, flatten=True )
    anno_names = CatmaidInterface.parse_annotation_list(anno_list,
                    output='names')
    return report_from_annotation_list( anno_names, CatmaidInterface )

def assert_pair( nrn_ids, pair_meta, CatmaidInterface ):
    """
        Use a id-based annotation with a pair-specifying meta-annotation to
        establish hemilateral pairs in CATMAID.

        While nrn_ids will usually be a pair, we have to account for the
        cases where there are multiple indistinguishable neurons (e.g.
        broad LNs). Naming order will be numerical, since this approahc doesn't
        have a unique left/right ordering.
    """

    # Check to see if neurons are already in a pair
    all_pair_annos = set(CatmaidInterface.get_annotations_from_meta_annotations(
                    pair_meta, flatten=True ) )
    nrn_annos = set( CatmaidInterface.get_annotations_for_objects(
                            nrn_ids ) )

    if len( all_pair_annos.intersection( nrn_annos ) ) > 0:
        for nrn_id in nrn_ids:
            specific_nrn_annos = set( CatmaidInterface.get_annotations_for_objects(
                                [ nrn_id ] ) )
            if len( specific_nrn_annos.intersection( all_pair_annos ) ) > 0:
                print( "Neuron with id {} is already paired!".format(nrn_id) )
    else:
        if len(nrn_ids) < 4:
            pair_name = 'hemilateral_pair' + ''.join(
                    ['_{}'.format(nid) for nid in sorted(nrn_ids)])
        else:
            pair_name = 'hemilateral_pair' + ''.join(
                    ['_{}'.format(nid) for nid in sorted(nrn_ids[0:4])])+'_etc'

        d = CatmaidInterface.add_annotation(annotation_list=[pair_name],
                                            id_list=nrn_ids,
                                            meta_list=pair_meta)
        if len(d['new_annotations'])==0:
            print('Warning! No new annotations created!')
        print(d['message'])
    return

def is_matched( nrn_ids, pair_meta, CatmaidInterface ):
    """
        is_matched( nrn_ids, pair_meta, CatmaidInterface)
    """
    annos_with_meta = CatmaidInterface.get_annotations_from_meta_annotations( pair_meta, flatten=True )
    ids_matched = CatmaidInterface.get_ids_from_annotations(annos_with_meta, flatten=True)
    has_match = {}
    for skid in nrn_ids:
        if skid in ids_matched:
            has_match[skid] = True
        else:
            has_match[skid] = False
    return has_match

def matched_complete_report( anno_list, pair_meta, CatmaidInterface, max_open_ends=0.03, min_nodes = 500 ):
    """

    """
    anno_names = []
    left_incom = []
    left_com = []
    left_incom_match = []
    left_com_match = []
    left_total = []

    right_incom = []
    right_com = []
    right_incom_match = []
    right_com_match = []
    right_total = []

    for anno in anno_list:
        if len(anno_list[anno]) < 2:
            continue
        else:
            print(anno)
        anno_names.append(anno)
        
        nrns_left_ids = CatmaidInterface.get_ids_from_annotations(
                                                        anno_list[anno]['l'],
                                                        flatten=True )
        
        is_matched_left = is_matched( nrns_left_ids, pair_meta, CatmaidInterface )
        props_left = property_summary_estimated( nrns_left_ids, CatmaidInterface )
        props_left['is_matched'] = pd.Series(is_matched_left)
        lincom, lincom_match, lcom, lcom_match = _match_category_helper(
                                                    props_left,
                                                    max_open_ends=max_open_ends,
                                                    min_nodes=min_nodes )

        left_incom.append( lincom )
        left_incom_match.append( lincom_match )
        left_com.append( lcom )
        left_com_match.append( lcom_match )
        left_total.append( lincom+lincom_match+lcom+lcom_match )
        
        nrns_right_ids = CatmaidInterface.get_ids_from_annotations(
                                                        anno_list[anno]['r'],
                                                        flatten=True )
        
        is_matched_right = is_matched( nrns_right_ids, pair_meta, CatmaidInterface )
        props_right = property_summary_estimated( nrns_right_ids, CatmaidInterface )
        props_right['is_matched'] = pd.Series(is_matched_right)
        rincom, rincom_match, rcom, rcom_match = _match_category_helper(
                                                    props_right,
                                                    max_open_ends=max_open_ends,
                                                    min_nodes=min_nodes )

        right_incom.append( rincom )
        right_incom_match.append( rincom_match )
        right_com.append( rcom )
        right_com_match.append( rcom_match )
        right_total.append( rincom+rincom_match+rcom+rcom_match )

    out = pd.DataFrame( {'Annotation':anno_names,
                         'Left Total':left_total,
                         'Left Unmatched Incomplete': left_incom,
                         'Left Matched Incomplete' : left_incom_match,
                         'Left Unmatched Complete' : left_com,
                         'Left Matched Complete' : left_com_match,
                         'Right Total':right_total,
                         'Right Unmatched Incomplete': right_incom,
                         'Right Matched Incomplete' : right_incom_match,
                         'Right Unmatched Complete' : right_com,
                         'Right Matched Complete' : right_com_match,
                         } )
    return out
        

def _match_category_helper( props, max_open_ends, min_nodes ):
    incom = sum(
            np.logical_and( np.logical_or( props['fraction_open_ends'] >= max_open_ends,
                                           props['node_count'] <= min_nodes ),
                            props['is_matched'] == False )
            )

    incom_match = sum(
            np.logical_and( np.logical_or( props['fraction_open_ends'] >= max_open_ends,
                                           props['node_count'] <= min_nodes ),
                            props['is_matched'] == True )
            )
    com = sum(
            np.logical_and( np.logical_and( props['fraction_open_ends'] < max_open_ends,
                                           props['node_count'] > min_nodes ),
                            props['is_matched'] == False )
            )
    com_match = sum(
            np.logical_and( np.logical_and( props['fraction_open_ends'] < max_open_ends,
                                           props['node_count'] > min_nodes ),
                            props['is_matched'] == True )
            )
    return incom, incom_match, com, com_match

def get_matched_id( skid, CatmaidInterface, pair_meta, include_self=False ):
    """

    """
    annos_with_meta = CatmaidInterface.get_annotations_from_meta_annotations( pair_meta, flatten=True )
    annos_for_skid = CatmaidInterface.get_annotations_for_objects( [skid] )
    pair_anno = list( set(annos_with_meta).intersection(annos_for_skid) )
    if len(pair_anno) > 0:
        ids_in_pair = CatmaidInterface.get_ids_from_annotations(pair_anno,flatten=True)
        if include_self:
            return list( set( ids_in_pair ) )
        else:
            return list( set( ids_in_pair ).difference(set([skid])) )
    else:
        print('No matched neuron!')
        return None


def filter_complete( id_list, CatmaidInterface, max_open_ends=0.03, min_node_count = 500, sensory_exception=False ):
    """

    """
    props = property_summary_estimated( id_list, CatmaidInterface )
    if sensory_exception:
        filt = (props['fraction_open_ends'] < max_open_ends) & (props['node_count'] > min_node_count)
    else:
        filt = (props['fraction_open_ends'] < max_open_ends) & (props['node_count'] > min_node_count) & ~(props['is_fragment'])

    return list( props[ filt ].index.values )

def _paired_ids_matched( CatmaidInterface,
                        match_report_df,
                        pair_meta,
                        max_open_ends=0.03,
                        min_node_count = 500 ):

    pair_list = []
    for row in match_report_df.iterrows():
        if isinstance( row[1]['Group_1'], np.integer ):
            rel_ids_1 = [row[1]['Group_1']]
        else:
            rel_ids_1 = row[1]['Group_1']

        if isinstance( row[1]['Group_2'], np.integer ):
            rel_ids_2 = [row[1]['Group_2']]
        else:
            rel_ids_2 = row[1]['Group_2']

        for id1 in rel_ids_1:
            for id2 in rel_ids_2:
                pair_list.append( [id1, id2] )
    return pair_list

def _paired_ids_unmatched_ipsilateral( CatmaidInterface,
                        id_list_1,
                        id_list_2,
                        pair_meta,
                        max_open_ends=0.03,
                        min_node_count = 500 ):

    pair_list = []
    for ind, id1 in enumerate(id_list_1):
        for id2 in id_list_1[ind+1:]:
            pair_list.append([id1, id2])

    for ind, id1 in enumerate(id_list_2):
        for id2 in id_list_2[ind+1:]:
            pair_list.append([id1, id2])
    return pair_list

def _paired_ids_unmatched_contralateral( CatmaidInterface,
                        id_list_1_comp,
                        id_list_2_comp,
                        match_report_df,
                        pair_meta,
                        max_open_ends=0.03,
                        min_node_count = 500 ):

    pair_list = []
    for row in match_report_df.iterrows():
        if isinstance( row[1]['Group_1'], np.integer ):
            match_ids_1 = [row[1]['Group_1']]
        else:
            match_ids_1 = row[1]['Group_1']
        non_match_ids_1 = list( 
                            set(id_list_1_comp).difference(set(match_ids_1))
                            )

        if isinstance( row[1]['Group_2'], np.integer ):
            match_ids_2 = [row[1]['Group_2']]
        else:
            match_ids_2 = row[1]['Group_2']
        non_match_ids_2 = list( 
                            set(id_list_2_comp).difference(set(match_ids_2))
                            )

        for id1 in match_ids_1:
            for id2 in non_match_ids_2:
                pair_list.append([id1,id2])

        for id2 in match_ids_2:
            for id1 in non_match_ids_1:
                pair_list.append([id1,id2])
    return pair_list

def make_id_pairs( CatmaidInterface,
                   id_list_1,
                   id_list_2,
                   pair_meta,
                   max_open_ends=0.03,
                   min_node_count = 500,
                   sensory_exception = False ):

    if type(id_list_1) is str:
        id_list_1 = CatmaidInterface.get_ids_from_annotations( id_list_1,
                                                               flatten=True )
    if type(id_list_2) is str:
        id_list_2 = CatmaidInterface.get_ids_from_annotations( id_list_2,
                                                               flatten=True )

    id_list_1_comp = filter_complete( id_list_1,
                                      CatmaidInterface,
                                      max_open_ends=max_open_ends,
                                      min_node_count=min_node_count,
                                      sensory_exception=sensory_exception)
    id_list_2_comp = filter_complete( id_list_2,
                                      CatmaidInterface,
                                      max_open_ends=max_open_ends,
                                      min_node_count=min_node_count,
                                      sensory_exception=sensory_exception)

    match_report_df = match_report( id_list_1_comp,
                                    id_list_2_comp,
                                    pair_meta,
                                    CatmaidInterface,
                                    name1='Group_1',
                                    name2='Group_2').drop('Unmatched')

    pairs_matched = _paired_ids_matched( CatmaidInterface,
                                        match_report_df,
                                        pair_meta,
                                        max_open_ends=max_open_ends,
                                        min_node_count=min_node_count)

    pairs_ipsi = _paired_ids_unmatched_ipsilateral( CatmaidInterface,
                                        id_list_1_comp,
                                        id_list_2_comp,
                                        pair_meta,
                                        max_open_ends=max_open_ends,
                                        min_node_count=min_node_count )

    pairs_contra = _paired_ids_unmatched_contralateral( CatmaidInterface,
                                        id_list_1_comp,
                                        id_list_2_comp,
                                        match_report_df,
                                        pair_meta,
                                        max_open_ends=max_open_ends,
                                        min_node_count=min_node_count )

    return pairs_matched, pairs_ipsi, pairs_contra


def match_groups_arbitrary( list_of_id_lists,
                            match_via,
                            CatmaidInterface,
                            nrns = None,
                            anno_reference='names' ):
    """
        Given N lists of neurons, create lists of groups based on a shared
        annotation (such as cell type) within a 'match_via' metaannotation.

        Parameters
        ----------
            id_list1 : list of list of ints
                List of skeleton ids in groups. Could be >2.

            match_via : string or int
                Annotation (as name or id) that annotates the annotations to match.

            CatmaidInterface : CatmaidDataInterface
                Interface for the Catmaid instance to query.

            nrns : NeuronListObject (default None)
                Optional NeuronListObject which already has neuron annotations
                listed. Must have all ids in list of id_lists to work.

            anno_reference : 'names' or 'ids' (optional, default is 'names')

        Returns
        -------
            dict
                Match report, indexed by annotation name or id.
    """
    annos_with_meta = set(CatmaidInterface.get_annotations_from_meta_annotations( match_via, flatten=True ) )
    annos_to_pair = []
    if nrns is None:
        for id_list in list_of_id_lists:
            annos_to_pair.append( { nid: set(CatmaidInterface.get_annotations_for_objects( [nid] )).intersection(annos_with_meta) for nid in id_list} )
    else:
        for id_list in list_of_id_lists:
            annos_to_pair.append( {nid: set( nrns[nid].annotations ).intersection(annos_with_meta) for nid in id_list} )

    matches = {}
    for group_ind, id_list in enumerate(list_of_id_lists):
        for nid in id_list:
            paired_annos = annos_to_pair[group_ind][nid]
            for anno in paired_annos:
                if anno not in matches:
                    matches[anno] = [[] for l in list_of_id_lists]
                matches[anno][group_ind].append( nid )

    return matches

