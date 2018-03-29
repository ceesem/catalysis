import catalysis.catmaid_interface as ci
import catalysis.neurons as na

from collections import defaultdict
import pandas as pd
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
        sn = CatmaidInterface.tag_query_for_skeleton( id, '^soma$|^to nerve$')
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
    annos_with_meta = set(CatmaidInterface.get_annotations_from_meta_annotations( match_via ) )
    annos1 = { id: set(CatmaidInterface.get_annotations_for_objects( [id] )).intersection(annos_with_meta) for id in id_list1}
    annos2 = { id: set(CatmaidInterface.get_annotations_for_objects( [id] )).intersection(annos_with_meta) for id in id_list2}

    matches = {}
    for id1 in annos1:
        for id2 in annos2:
            for anno_id in annos1[id1].intersection(annos2[id2]):
                if anno_reference is 'ids':
                    matches[anno_id] = [id1, id2]
                elif anno_reference is 'names':
                    matches[ CatmaidInterface.parse_annotation_list(anno_id, output='names') ] = [id1, id2]

    return matches

def match_report( id_list1, id_list2, match_via, CatmaidInterface, name1='Group 1', name2='Group 2', anno_reference = 'names' ):
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

            Returns
            -------
                DataFrame
                    Organized, readable match report

    """
    annos = CatmaidInterface.get_annotations()
    rev_dict = { annos[key] : key for key in annos }

    matches = match_groups( id_list1, id_list2, match_via, CatmaidInterface )
    matched = [[],[]]
    match_report1 = {}
    match_report2 = {}
    for anno_id in matches:
        #anno_name = annos[anno_id] + ' (' + str(anno_id) + ')'
        anno_name = rev_dict[anno_id] + ' (' + str(anno_id) + ')'
        matched[0].append( matches[anno_id][0] )
        matched[1].append( matches[anno_id][1] )
        match_report1[ anno_name ] = matches[anno_id][0]
        match_report2[ anno_name ] = matches[anno_id][1]

    unmatched1 = set(id_list1).difference( set(match_report1.values()) )
    unmatched2 = set(id_list2).difference( set(match_report2.values()) )
    match_report1['Unmatched'] = list(unmatched1)
    match_report2['Unmatched'] = list(unmatched2)

    print( str( len(set(matched[0])) ) + ' of ' + str(len(id_list1)) + ' of ' + name1 + ' matched...')
    print( str( len(set(matched[1])) ) + ' of ' + str(len(id_list2)) + ' of ' + name2 +  ' matched...')

    report = pd.DataFrame( { name1: match_report1, name2: match_report2} )
    return report

def match_report_from_annos( anno1, anno2, match_via, CatmaidInterface, anno_reference = 'names'):
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
                Annotation (as name or id) that annotates the annotatoins to
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
                         name1 = anno1,
                         name2 = anno2)

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