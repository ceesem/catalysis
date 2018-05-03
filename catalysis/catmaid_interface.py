# -*- coding: utf-8 -*-

from catpy import *
from itertools import chain
import networkx as nx
import numpy as np
from IPython.core.display import display, HTML


class CatmaidDataInterface():
    """
        Python object using cat.CatmaidClient to aid interaction with neuronal circuit data.
    """

    def __init__(self, CatmaidClient):
        """
            Instantiate a data interface with a CatmaidClient object.

            Parameters
            ----------
            CatmaidClient : catpy.client.CatmaidClient object
                Object that hosts parameters to talk to a CATMAID instance through its API

            Returns
            ----------
        """
        self.CatmaidClient = CatmaidClient

    @classmethod
    def from_json( cls, filename ):
        return cls( client.CatmaidClient.from_json( filename ) )

    def skip_ids( self, id_list, ids_to_skip ):
        """
            Simple helper function to strip certain ids from a list.
        """
        return list( set(id_list).difference( set(ids_to_skip) ) )

    def set_difference_annotations( self, id_list, skip_annotations ):

        ids_to_skip = self.get_ids_from_annotations(skip_annotations, flatten=True)
        return self.skip_ids(id_list, ids_to_skip)

    def url_to_point( self, xyz, node_skeleton_ids = None, tool = 'tracingtool', zoomlevel=0, show=True):
        """
            Generate a URL that goes to a specified location (and, optionally, a selected node) in a CATMAID instance.

            Parameters
            ----------
            xyz : 3-element list or tuple of ints
                Species the x, y, and z location (at indices 0, 1, and 2 respectively) of the target point in the CATMAID volume.
            node_skeleton_ids: 2-element list or tuple of ints, optional
                Tuple or list of a node id and skeleton id to make selected, ordered as (node_id, skeleton_id).
            tool: str, optional
                Defines the CATMAID widget to be active. Defaults to the tracing tool.
            zoomlevel: int, optional
                Defines the zoom level of the CATMAID view. Defaults to 0.

            Returns
            ----------
            str
                URL that goes to the CATMAID instance, centered on the view specified and with an optional selected node.

        """
        base_url = self.CatmaidClient.base_url + '/?'
        pid_str = 'pid=' + str(self.CatmaidClient.project_id)
        x_str = 'xp=' + str( xyz[0] )
        y_str = 'yp=' + str( xyz[1] )
        z_str = 'zp=' + str( xyz[2] )
        tool_str = 'tool=' + tool
        sid_str = 'sid0=1'
        zoom_str = 's0=' + str( zoomlevel )
        if node_skeleton_ids is not None:
            node_str = 'active_node_id=' + str( node_skeleton_ids[0] )
            skid_str = 'active_skeleton_id=' + str( node_skeleton_ids[1] )
            strs = [base_url,pid_str,x_str,y_str,z_str,node_str,skid_str,tool_str,sid_str,zoom_str]
        else:
            strs = [base_url,pid_str,x_str,y_str,z_str,tool_str,sid_str,zoom_str]
        out_url = '&'.join(strs)
        if show==True:
            display(HTML(
                '<a href="{}">{}</a>'.format(
                                out_url,
                                'Link to point '+str(xyz) ) ) )
        return out_url

    def url_to_neurons( self, id_list ):
        """
            Generate CATMAID url to the neuron's root for each neuron in an iterable
            id list.
        """
        if isinstance(id_list, (int, np.integer)):
            id_list = [id_list]

        names = self.get_neuron_names(id_list)

        for neuron_id in id_list:
            root_info = self.root_for_skeleton( neuron_id )
            xyz = [root_info['x'], root_info['y'], root_info['z']]
            display(HTML(
                '<a href="{}">{}</a>'.format(
                                self.url_to_point( xyz,
                                                  (root_info['root_id'], neuron_id)
                                                  ),
                                names[str(neuron_id)] )
                ))
        return None


    def node_location( self, node_id ):
        """
    
        """
        url = '/{}/treenodes/{}/compact-detail'.format(self.CatmaidClient.project_id,
                                                       node_id )
        d = self.CatmaidClient.get(url)
        xyz = [d['x'],d['y'],d['z']]
        return xyz

    def get_neuron_name( self, skeleton_id ):
        """
            Get the name of a single neuron from its skeleton id.

            Parameters
            ----------
            skeleton_id : int
                Skeleton id for the neuron whose name is to be retrieved.

            Returns
            ----------
            str
                Name of neuron
        """
        d = self.CatmaidClient.get( '/{}/skeleton/{}/neuronname'.format( self.CatmaidClient.project_id, skeleton_id) )
        if d.get('neuronname') is None:
            print('No neuron with skeleton id ' + str(skeleton_id) )
        return d.get('neuronname')

    def get_review_status( self, id_list, user_whitelist=None ):
        url = '/{}/skeletons/review-status'.format(self.CatmaidClient.project_id)
        postdata = {}
        for ind, id in enumerate(id_list):
            postdata[ 'skeleton_ids[{}]'.format(ind) ] = id
        if user_whitelist is not None:
            for ind, uid in enumerate(user_whitelist):
                postdata[ 'user_ids[{}]'.format(ind)] = uid
        return self.CatmaidClient.post( url, data = postdata)

    def root_for_skeleton( self, id ):
        """
            Returns the root node of a given skeleton id. Direct version of API.
        """
        url = '/{}/skeletons/{}/root'.format( self.CatmaidClient.project_id, id)
        return self.CatmaidClient.get( url )

    def tag_query_for_skeleton( self, id, tag_query ):
        """
            Find nodes on a skeleton with a tag corresponding to a regex search on tag_query
        """
        url = '/{}/skeletons/{}/find-labels'.format( self.CatmaidClient.project_id, id)
        data = { 'treenode_id': self.root_for_skeleton(id)['root_id'],
                 'label_regex' : tag_query}
        return self.CatmaidClient.post( url, data=data )

    def get_closed_ends( self, id ):
        """
            Gets nodes for a skeleton with the blessed tags that signal a true end. Might not actually be an end.
        """
        tag_query = '^ends$|^soma$|^uncertain end$|^uncertain continuation$|^not a branch$'
        return self.tag_query_for_skeleton( id, tag_query=tag_query )

    def get_open_ends( self, id ):
        """
            Retrieve open (untagged with a blessed completion tag) end nodes for a skeleton id.
        """
        url = '/{}/skeletons/{}/open-leaves'.format( self.CatmaidClient.project_id, id)
        data = {'treenode_id': self.root_for_skeleton( id )['root_id'] }
        return self.CatmaidClient.post( url, data=data )

    def node_count( self, id):
        """

        """
        url = '/{}/skeleton/{}/node_count'.format( self.CatmaidClient.project_id, id)
        # data = {'treenode_id': self.root_for_skeleton( id )['root_id'] }
        return self.CatmaidClient.post( url )

    def get_neuron_names( self, id_list ):
        """
            Get names of a list of neurons from their skeleton ids.

            Parameters
            ----------
            id_list : list of ints
                List of skeleton ids for which to return neurons

            Returns
            ----------
            Dict
                Dict of skeleton ids and associated names, {id:name}.
        """
        url = '/{}/skeleton/neuronnames'.format( self.CatmaidClient.project_id )
        postdata = {}
        for ind, id in enumerate(id_list):
            postdata[ 'skids[{}]'.format(ind) ] = id

        return self.CatmaidClient.post( url, data=postdata )

    def _get_compact_skeleton( self, skeleton_id, withtags ):
        """
            Internal function for using compact-skeleton to get skeleton data
        """
        if withtags:
            tag_flag = 1
        else:
            tag_flag = 0
        url = '/{}/{}/1/{}/compact-skeleton'.format( self.CatmaidClient.project_id, skeleton_id, tag_flag)
        return self.CatmaidClient.get( url )

    def get_skeleton_data( self,
                           skeleton_id_list,
                           with_tags=True,
                           with_connectors=True,
                           with_history=False,
                           with_merge_history=False,
                           with_reviews=False,
                           with_annotations=False,
                           with_user_info=False,
                           max_neurons_per_post=50):
        """
        """
        
        id_iter = iter(skeleton_id_list)
        skeleton_lists = []
        
        ind = 0
        finished = False
        while not finished:
            skeleton_lists.append([])
            for ii in np.arange(0,max_neurons_per_post):
                try:
                    skeleton_lists[ind].append( next(id_iter) )
                except:
                    finished=True
                    break
            ind+=1
        
        returned_data = []
        for id_list in skeleton_lists:
            postdata = {}

            for ind, id in enumerate(id_list):
                postdata[ 'skeleton_ids[{}]'.format(ind) ] = id

            if with_tags:
                postdata['with_tags'] = 'true'
            else:
                postdata['with_tags'] = 'false'

            if with_connectors:
                postdata['with_connectors'] = 'true'
            else:
                postdata['with_connectors'] = 'false'

            if with_history:
                postdata['with_history'] = 'true'
            else:
                postdata['with_history'] = 'false'

            if with_merge_history:
                postdata['with_merge_history'] = 'true'
            else:
                postdata['with_merge_history'] = 'false'

            if with_reviews:
                postdata['with_reviews'] = 'true'
            else:
                postdata['with_reviews'] = 'false'

            if with_annotations:
                postdata['with_annotations'] = 'true'
            else:
                postdata['with_annotations'] = 'false'

            if with_user_info:
                postdata['with_user_info'] = 'true'
            else:
                postdata['with_user_info'] = 'false'

            url = '/{}/skeletons/compact-detail'.format( self.CatmaidClient.project_id )
            returned_data.append( self.CatmaidClient.post( url, data = postdata ) )

        out = returned_data[0]
        for d in returned_data[1:]:
            out['skeletons'].update(d['skeletons'])
        return out

    def skeleton_statistics( self, skeleton_id ):
        """
            Get large-scale skeleton statistics
        """
        url = '{}/skeleton/{}/statistics'.format(self.CatmaidClient.project_id, skeleton_id)
        d = self.CatmaidClient.get(url)
        return d

    def total_inputs( self, id_list ):
        """
            Get large-scale skeleton statistics
        """
        d = self._get_connected_skeleton_info(id_list)
        total_inputs = {skid:0 for skid in id_list}
        for pid in d['incoming']:
            for skid in d['incoming'][pid]['skids']:
                total_inputs[int(skid)] += sum( d['incoming'][pid]['skids'][skid] )
        return total_inputs

    def postsynaptic_count( self, connector_list ):
        """
            Get the number of postsynaptic targets for a list of connector ids.

            Parameters
            ----------
            connector_list: list of ints
                Iterable list of connector ids

            Returns
            ----------
            dict
                Dict with connector ids as keys (ints) and the number of postsynaptic targets (ints) as values.
        """
        nps = dict()
        if len( connector_list ) > 0:
            url = '/{}/connector/skeletons'.format(self.CatmaidClient.project_id)
            opts = {}
            for ind, id in enumerate(connector_list):
                opts[ 'connector_ids[{}]'.format(ind) ] = id
            d = self.CatmaidClient.post( url, data = opts )
            for conn in d:
                nps[conn[0]] = len( conn[1]['postsynaptic_to'] )
        return nps

    def get_connector_info( self, connector_id ):
        """
            Gets information about the location and connections for a connector

            Parameters
            ----------
            connector_id : int
                Id of the connector to look up

            Returns
            ----------
            dict
                Dict with various information about the connector.
        """
        url = '/{}/connectors/{}/'.format( self.CatmaidClient.project_id, connector_id )
        return self.CatmaidClient.get( url, force_json=True )

    def get_connector_data( self, connector_id_list ):
        """
            Gets information about the location and connections for a connector

            Parameters
            ----------
            connector_id_list : List of ints
                Ids of connectors to look up

            Returns
            ----------
            dict
                Dict with information about the connectors.
        """
        if len( connector_id_list ) > 0:
            url = '/{}/connector/skeletons'.format( self.CatmaidClient.project_id )
            opts = {}
            for ind, connid in enumerate(connector_id_list):
                opts[ 'connector_ids[{}]'.format(ind) ] = connid

            d = self.CatmaidClient.post( url, data = opts, force_json = True )
        else:
            d = []
        return d

    def get_cable_length( self, id_list, num_per_call = 50 ):
        """ 
            Get cable length for a list of skeleton ids.
        """
        url = '/{}/skeletons/cable-length'.format( self.CatmaidClient.project_id )
        data = {}
        
        id_list_chunked = [ id_list[ii:ii+num_per_call] for ii in range(0, len(id_list), num_per_call)]
        lens = {}
        for chunk in id_list_chunked:
            for ind, skid in enumerate(chunk):
                data['skeleton_ids[{}]'.format( ind ) ] = skid
                lens.update( self.CatmaidClient.get( url, params=data ) )
        return self.CatmaidClient.get( url, params=data )

    def get_annotations( self ):
        """
            Retrieve the list of annotations and the annotation ids.

            Parameters
            ----------
            None

            Returns
            ----------
                dict
                    Dict with annotation string as keys, annotation id as values.
        """
        url = '/{}/annotations/'.format( self.CatmaidClient.project_id )
        all_annotations = self.CatmaidClient.get( url )
        anno_dict = { item['name']:item['id'] for item in all_annotations['annotations'] }
        return anno_dict

    def parse_annotation_list( self, anno_list, output = 'ids' ):
        """
            Take a list of annotations, whether names or ids, and get back a list of annotation ids.

            Parameters
            ----------
                anno_list : list of strings or ints (or a single annotation)
                    Some format for annotations, whether list or a scalar, ids or strings.

                output : String
                    Selects which type of output is desired. 'ids' if string ids, 'names' if the annotation text.

            Returns
            -------
                list
                    List of annotations, either as ids or strings depending on the to_ids parameter.
        """
        if type(anno_list) is str or type(anno_list) is int:
            anno_list = [anno_list]

        anno_dict_names = self.get_annotations()
        if output is 'names':
            anno_dict_ids = { anno_dict_names[name]:name for name in anno_dict_names }

        annotation_output = []
        if output is 'ids':
            for anno in anno_list:
                if type(anno) is int:
                    annotation_output.append(anno)
                elif type(anno) is str:
                    try:
                        annotation_output.append(anno_dict_names[anno])
                    except KeyError:
                        print( "\"", anno, "\" is not a valid annotation" )

        elif output is 'names':
            for anno in anno_list:
                if type(anno) is int:
                    try:
                        annotation_output.append(anno_dict_ids[anno])
                    except KeyError:
                        print( "\"", anno, "\" is not a valid annotation" )

                elif type(anno) is str:
                    annotation_output.append(anno)
        # else:
        #     raise(ValueError,'Output must be either \'ids\' or \'names\'')
        return annotation_output


    def get_annotation_ids_from_names( self, annotation_list ):
        """
            Return the annotation ids from a list of annotations as strings

            Parameters
            ----------
            annotation_list : list of strings
                List of annotations to query for.

            Returns
            ----------
            list of ints
                List of annotation ids corresponding to the string list.
        """
        if type(annotation_list) is str:
            annotation_list = [annotation_list]

        anno_dict = self.get_annotations()
        annotation_id_list = []

        for anno in annotation_list:
            try:
                annotation_id_list.append( anno_dict[anno] )
            except KeyError:
                print( "\"", anno, "\" is not a valid annotation" )

        return annotation_id_list

    def get_annotations_for_objects( self, id_list ):
            url = '/{}/annotations/forskeletons'.format( self.CatmaidClient.project_id )
            opts = {}
            for ind, skid in enumerate(id_list):
                opts[ 'skeleton_ids[{}]'.format( ind ) ] = skid
            d = self.CatmaidClient.post( url, data=opts )
            if len(d) > 0:
                return [int(id) for id in d['annotations']]
            else:
                return None

    def get_annotations_from_meta_annotations( self, meta_list, flatten = False):
        """
        """
        meta_id_list = self.parse_annotation_list( meta_list )
        return self.get_annotations_from_meta_annotation_ids( meta_id_list, flatten )

    def get_annotations_from_meta_annotation_ids( self, meta_id_list, flatten=False ):
        """
            Get the annotation ids annotated with a list of metaannotation ids.

            Parameters
            ----------
            meta_id_list : list of ints
                Annotation ids to pull metaannotations for.

            Returns
            ----------
            dict (default, if flatten = False )
                dict of lists of skeleton ids

            list (if flatten = True)
                List of skeleton ids corresponding to any of the annotations.
        """
        if type(meta_id_list) is not list:
            meta_id_list = [ meta_id_list ]
        url = '/{}/annotations/query-targets'.format( self.CatmaidClient.project_id )

        meta_to_anno_ids = {}
        for anno_id in meta_id_list:
            d = self.CatmaidClient.post( url, data = { 'annotated_with' : anno_id } )
            out = [ item['id'] for item in d['entities'] if item['type'] == 'annotation'  ]
            meta_to_anno_ids[anno_id] = out

        if flatten:
            all_ids = [skid for skid in chain.from_iterable( meta_to_anno_ids.values() )]
            return list( set( all_ids ) )
        else:
            return meta_to_anno_ids

    def get_annotation_names( self, annotation_id_list, flatten=False):
            """
                Get the names of annotations cooresponding to ids.

                Parameters
                ----------
                annotation_id_list : list of ints
                    Annotation ids to find names for.

                flatten : Boolean (default is false)
                    Parameter determing if response is a list (True) or a dict (False)

                Returns
                -------
                list or dict
                    List or dict containing the names. If a dict, keys are annotation ids.
            """
            anno_names = self.parse_annotation_list( annotation_id_list, output = 'names ')
            if flatten:
                return anno_names
            else:
                return { id:anno_names[ind] for ind, id in enumerate(annotation_id_list) }

    def get_ids_from_annotations( self, annotation_id_list, skip_annotations=None, flatten = False ):
        """
            Get the skeleton ids annotated with a list of annotation ids.

            Parameters
            ----------
            annotation_id_list : list of ints or names
                Annotation ids or names to pull skeletons for (or even mix of the two)

            Returns
            ----------
            dict (default, if flatten = False )
                dict of lists of skeleton ids

            list (if flatten = True)
                List of skeleton ids corresponding to any of the annotations.
        """

        annotation_id_list = self.parse_annotation_list(annotation_id_list, output = 'ids' )

        if skip_annotations is not None:
            ids_to_skip = self.get_ids_from_annotations(skip_annotations)
        else:
            ids_to_skip = []

        url = '/{}/annotations/query-targets'.format( self.CatmaidClient.project_id )

        anno_to_skids = {}
        for anno_id in annotation_id_list:
            d = self.CatmaidClient.post( url, data = { 'annotated_with' : anno_id } )
            ids_returned = [ item['skeleton_ids'][0] for item in d['entities'] ]
            anno_to_skids[anno_id] = self.skip_ids( ids_returned, ids_to_skip )

        if flatten:
            all_ids = [skid for skid in chain.from_iterable( anno_to_skids.values() )]
            return list( set( all_ids ) )
        else:
            return anno_to_skids

    def add_annotation( self, annotation_list, id_list, meta_list=None ):
        """
            Add annotations to neurons in the CATMAID database.

            Parameters
            ----------
            annotation_list : list of strings (or just one)
                A list of annotations to be applied to specified neurons.

            id_list : list of ints
                Skeleton ids to which to add annotations.
            
            meta_list : list of strings
                Meta-annotations to add to annotations in annotation_list.

            Returns
            ----------
            dict
                Contains information about the success and novelty of annotations.

        """
        if type(annotation_list) is str:
            annotation_list = [annotation_list]
        if type(id_list) is int:
            id_list = [id_list]
        if type(meta_list) is str:
            meta_list = [meta_list]

        url = '/{}/annotations/add'.format( self.CatmaidClient.project_id )
        postdata = dict()
        for i, anno in enumerate(annotation_list):
            postdata['annotations[{}]'.format(i)] = anno
        for i, id in enumerate(id_list):
            postdata['skeleton_ids[{}]'.format(i)] = id
        if meta_list is not None:
            for i, id in enumerate(meta_list):
                postdata['meta_annotations[{}]'.format(i)] = id
        d = self.CatmaidClient.post( url, data=postdata )
        return d

    def in_surrounding_box( self, xyz0, d ):
        """
            Get skeleton ids within a bounding box evenly surrounding a point with distance d.
            Parameters
            ----------

                xyz0 : 3 element array-like
                    Location of the center of the box in x,y,z

                d : number
                    Distance from center point to locate the bounding box walls

            Returns
            -------
                list
                    skeleton ids passing through the box. 
        """

        return self.in_bounding_box(
                                    xyz0[0]-d,
                                    xyz0[0]+d,
                                    xyz0[1]-d,
                                    xyz0[1]+d,
                                    xyz0[2]-d,
                                    xyz0[2]+d )

    def in_bounding_box( self, x_min, x_max, y_min, y_max, z_min, z_max ):
        """
            Get skeleton in within a bounding box specified in world coordinates.
            
            Parameters
            ----------
                x_min : number
                x_max : number
                y_min : number
                y_max : number
                z_min : number
                z_max : number
                    Each number corresponds to the of the extremes of the bounding box.

            Returns
            -------

            list
                List of skeleton ids passing through the bounding box.

        """
        url = '/{}/skeletons/in-bounding-box'.format( self.CatmaidClient.project_id )
        data = {
            'minx':x_min,
            'miny':y_min,
            'minz':z_min,
            'maxx':x_max,
            'maxy':y_max,
            'maxz':z_max
        }
        return self.CatmaidClient.get( url, params=data )

    def _get_connected_skeleton_info( self, id_list  ):
        """
        Retreives the skeleton info (node count, number of synapses, etc) for neurons to a supplied list of neurons.

        Parameters
        ----------
        id_list : list of ints
            A list of skeleton ids whose connectivity is to be looked up.

        Returns
        ----------
        dict
            The first key is 'incoming' or 'outgoing', the second key is the skeleton id (int) of a connected neuron,
            the third key is 'num_nodes' (returns number of nodes in a skeleton) or 'skids', the fourth key (if skids) is a skeleton id (string) of a supplied neuron, and the value is a list of number of synapses
            indexed by confidence (5 entries).

        """
        if type(id_list) is int:
            id_list = [id_list]

        url = '/{}/skeletons/connectivity'.format( self.CatmaidClient.project_id )
        postdata = {'boolean_op' : 'OR'}
        for i, id in enumerate( id_list ):
            postdata['source_skeleton_ids[{}]'.format(i)] = id
        d = self.CatmaidClient.post( url, data=postdata )

        connected_info = dict()
        connected_info['incoming'] = {}
        connected_info['outgoing'] = {}
        for id in d['incoming'].keys():
            connected_info['incoming'][int(id)] = d['incoming'][id]
        for id in d['outgoing'].keys():
            connected_info['outgoing'][int(id)] = d['outgoing'][id]
        return connected_info

    def get_connected_skeletons( self, id_list, include_presynaptic = True, include_postsynaptic = True, presynaptic_cutoff = 0, postsynaptic_cutoff = 0, node_cutoff = 0):
        """
            Get a list of all neurons and synaptic counts for a list of skeleton ids.

            Parameters
            ----------
            id_list : list of ints
                Id list for which to retrieve connectivity

            include_presynaptic : Bool (optional, default is True)
                Boolean value indicating whether to include neurons presynaptic to those specified.

            include_postsynaptic : Bool (optional, default is True)
                Boolean value indicating whether to include neurons postsynaptic to those specified.

            presynaptic_cutoff : Int (optional, default is 0)
                Minimum number of synapses for presynaptic neurons to return

            postsynaptic_cutoff : Int (optional, default is 0)
                Minimum number of synapses for postsynaptic neurons to return

            node_cutoff : Int (optinal, default is 0)
                Minimum number of nodes in skeletons to return

            Returns
            ----------
            dict

        """

        connected_info = self._get_connected_skeleton_info( id_list )
        output_skeletons = {}

        if include_presynaptic:
            pre_skeletons = { skid: {} for skid in id_list }
            for pre_id in connected_info['incoming']:
                if connected_info['incoming'][pre_id]['num_nodes'] > node_cutoff:
                    for skid in connected_info['incoming'][pre_id]['skids']:
                        n_syn = sum(connected_info['incoming'][pre_id]['skids'][skid])
                        if n_syn > presynaptic_cutoff:
                            pre_skeletons[int(skid)][pre_id] = n_syn
                else:
                    continue
            output_skeletons['presynaptic'] = pre_skeletons
        else:
            output_skeletons['presynaptic'] = []

        if include_postsynaptic:
            post_skeletons = { skid: {} for skid in id_list }
            for post_id in connected_info['outgoing']:
                if connected_info['outgoing'][post_id]['num_nodes'] > node_cutoff:
                    for skid in connected_info['outgoing'][post_id]['skids']:
                        n_syn = sum(connected_info['outgoing'][post_id]['skids'][skid])
                        if n_syn > postsynaptic_cutoff:
                            post_skeletons[int(skid)][post_id] = n_syn
                else:
                    continue
            output_skeletons['postsynaptic'] = post_skeletons
        else:
            output_skeletons['postsynaptic'] = []

        return output_skeletons

    def get_connected_skeletons_flat( self, id_list, include_presynaptic = True, include_postsynaptic = True, presynaptic_cutoff = 0, postsynaptic_cutoff = 0, node_cutoff = 0):
        conn_sk = self.get_connected_skeletons( id_list, include_presynaptic, include_postsynaptic, presynaptic_cutoff, postsynaptic_cutoff, node_cutoff )
        id_list = []
        for direction in conn_sk:
            for skid in conn_sk[direction]:
                id_list += list(conn_sk[direction][skid].keys())
        return list(set(id_list))

    def get_connectivity_graph_edge_list( self, id_list ):
        """
            Retrieve the graph of neurons as an edge list weighted by synaptic count

            Parameters
            ----------
            id_list : List of ints
                List of neuron ids to find graph for

            Returns
            ----------
            List
                Each row in the list is an edge: [presynaptic id, postsynaptic id, #synapses]
        """
        url = '/{}/skeletons/confidence-compartment-subgraph'.format( self.CatmaidClient.project_id )
        postdata = {}
        for i, id in enumerate(id_list):
            postdata['skeleton_ids[{}]'.format(i)] = id
        d = self.CatmaidClient.post( url, data=postdata )
        edges = []
        for row in d['edges']:
            edges.append( [row[0],row[1], sum(row[2] )] )

        return edges

    def get_network_neighborhood( self, id_list, number_of_hops, pre_minimum_synapses=1, post_minimum_synapses=1, include_names = False ):
        """
            Find ids connected to a list of neurons within a certain number of hops

            Parameters
            ----------
            id_list : List of ints
                List of skeleton ids of neurons that serve as the core.

            number_of_hops : int
                Number of hops away from the core neurons to include. Must be larger than 1.

            pre_minimum_synapses : int (optional, default=1)
                Minimum number of synapses to traverse presynaptic to core neurons. Use None to not traverse presynaptic at all.

            post_minimum_synapses : int (optional, default=1)
                Minimum number of synapses to traverse postsynaptic to core neurons. Use None to not traverse postsynaptic at all.

            include_names : Bool (optional, default = False)
                If True, return a dict of id:names. If false, just return a list of ids.

            Returns
            ----------
            List (if include_names = False, default)
                List of skeleton ids matching the requested network.

            Dict (if include_names = True)
                Dict of {skeleton ids: name} matching the requested network.
        """
        url = '/{}/graph/circlesofhell'.format( self.CatmaidClient.project_id)
        postdata = {}
        postdata['n_circles'] = number_of_hops
        postdata['min_pre'] = pre_minimum_synapses
        postdata['min_post'] = post_minimum_synapses
        for i, id in enumerate(id_list):
            postdata['skeleton_ids[{}]'.format(i)] = id
        d = self.CatmaidClient.post( url, data=postdata )
        if include_names is False:
            return d[0]
        else:
            return d[1]

    def find_intermediate_neurons( self, origin_ids, target_ids, number_of_hops, minimum_synapses=1, include_ends = False ):
        """
            Find neurons along paths between origins and targets.

            Parameters
            ----------
            origin_ids : list of ints
                Skeleton ids from which to begin all paths (presynaptic).

            target_ids : list of ints
                Skeleton ids from which to end all paths (postsynaptic).

            number_of_hops : int
                Number of hops in the connections to query.

            minimum_synapses : int (optional, default = 1)
                Minimum number of synapses in a given hop.

            include_ends : Bool (optional, default = False)
                Boolean option to return skeleton ids of the origin/target ids or just the intermediate neuron ids.

            Returns
            -------
            list
                List of neurons along the types of path specified.
        """

        nodes_on_paths = set()
        for o_hops in range(1,number_of_hops-1):
            t_hops = number_of_hops - o_hops

            # From origin ids, only expand postsynaptically.
            origin_neighborhood = set( self.get_network_neighborhood(
                                        origin_ids,
                                        o_hops,
                                        pre_minimum_synapses = None,
                                        post_minimum_synapses = minimum_synapses,
                                        ) )

            # From target ids, only expand presynaptically.
            target_neighborhood = set( self.get_network_neighborhood(
                                        target_ids,
                                        t_hops,
                                        pre_minimum_synapses = minimum_synapses,
                                        post_minimum_synapses = None,
                                        ) )

            nodes_on_paths.update( origin_neighborhood.intersection( target_neighborhood ) )

        if include_ends is True:
            nodes_on_paths = nodes_on_paths.union(origin_ids)
            nodes_on_paths = nodes_on_paths.union(target_ids)

        return list(nodes_on_paths)

    def neuron_graph_from_ids( self, id_list ):
        """
            Build a networkx graph of neuronal connectivity from a list of neuron ids.

            Parameters
            ----------
            id_list : List of ints
                Iterable list of skeleton ids for which to generate a graph.

            CatmaidInterface : CatmaidDataInterface
                Object initialized to a Catmaid project.

            Returns
            ----------
            NetworkX DiGraph
                Graph with keys being neuron ids and number of synapses being the edge attribute 'weight'. Nodes also have attribute "name".
        """
        g = nx.DiGraph()

        for id in id_list:
            g.add_node(id)
            g.node[id]['name'] = self.get_neuron_name( id )

        cm_edges = self.get_connectivity_graph_edge_list( id_list )
        for e in cm_edges:
            g.add_edge( e[0], e[1], weight = e[2] )

        return g

    def get_landmarks( self, with_locations=True ):
        """
            Get all landmarks from CatmaidInstance
        """
        url = '/{}/landmarks/?with_locations={}'.format(
                    self.CatmaidClient.project_id,
                    str(with_locations).lower() )
        return self.CatmaidClient.fetch( url )

    def get_landmark_groups( self, with_locations=False, with_members=False ):
        """
            Get landmark groups (set of landmarks that correspond to a category like left or right)
            from the CatmaidInstance
        """
        url = '/{}/landmarks/groups/?with_locations={}&with_members={}'.format(
                    self.CatmaidClient.project_id,
                    str(with_locations).lower(),
                    str(with_members).lower() )
        return self.CatmaidClient.fetch( url )

    def match_groups_from_select_annotations( self, annotation_match, group_parser):
        """
            Use regex to select a set of annotations that correspond to repeated instances
            of a category of neurons and then parse the different instances into groups.
            For example, look at brain annotations and group them into left and right, 
            or VNC annotations and group into segment.

            Parameters
            ----------
            annotation_match : regex compile object
                Regex object that matches on annotations to be considered (e.g. lineage names)

            group_parser : regex compile object
                Regex object that finds two strings, a common group name and a subgroup identifier.
                One captured item must be called 'group' and the other 'instance'.
                For example, re.compile('\*(?P<group>.*?)_(?P<instance>[rl]) akira') would find a core
                lineage name and an r/l instance in a string like '*BAla12_r akira'

            Returns 

        """

        select_annotations = []

        annos = self.get_annotations()
        for anno in annos:
            if annotation_match.match(anno):
                select_annotations.append(anno)
        select_annotations = sorted(select_annotations)

        groups = {}
        
        for anno in select_annotations:
            group_name = group_parser.match(anno).groupdict()['group']
            instance_name = group_parser.match(anno).groupdict()['instance']

            if group_name not in groups:
                groups[group_name] = {}
            groups[group_name][instance_name] = anno

        return groups


    def reroot_neurons_to_soma( self, id_list ):
        """
        Run through an id_list and make sure that,if there is a single soma, the neuron
        is rooted to it.
        """
        for skid in id_list:
            tag_query = '^soma$|^cell body'
            dsoma = self.tag_query_for_skeleton( skid, tag_query=tag_query )
            if len(dsoma) == 0:
                print('Skipping {} for want of a soma. Fix at the link below:'.format(skid))
                self.url_to_neurons([skid])

                continue
            elif len(dsoma) > 1:
                print('Skipping {} which has too many somata. Fix at the link below:'.format(skid))
                self.url_to_neurons([skid])
                continue
            else:
                curr_root = self.root_for_skeleton( skid )
                if curr_root['root_id'] != dsoma[0][0]:
                    url = '/{}/skeleton/reroot'.format( self.CatmaidClient.project_id )
                    dat = {'treenode_id': dsoma[0][0]}
                    d = self.CatmaidClient.post( url, data=dat )
                    print('Rerooted {}...'.format(skid))