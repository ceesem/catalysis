import catalysis.catmaid_interface as ci
import re
import scipy as sp
import numpy as np
from scipy import spatial
import scipy.sparse.csgraph as csgraph
import scipy.sparse as sparse
from collections import defaultdict
import copy
import json
import datetime
import csv
import pandas as pd
import gzip
import copy
from itertools import chain

try:
    import cPickle as pickle
except:
    import pickle

class NeuronList:
    def __init__( self, NeuronObjDict, CatmaidInterface = None, export_date = None, project_name = None ):
        """
            Initialize a NeuronList object comprised of neurons, the catmaid interface object that created them.

            Parameters
            ----------
            NeuronObjDict : Dict of NeuronObj
                Dict of neurons keyed by unique id

            CatmaidInterface : CatmaidDataInterface (optional, default = None)
                CatmaidInterface for future querying if allowed.
        """
        self.neurons = NeuronObjDict
        self.CatmaidInterface = CatmaidInterface
        self.export_date = export_date
        self.project_name = project_name

    def __getitem__( self, key ):
        try:
            out = self.neurons[key]
        except TypeError:
            raise TypeError
        except IndexError:
            raise IndexError
        except KeyError:
            raise KeyError
        return out

    def __len__( self ):
        return len(self.neurons)

    def __iter__(self):
        for neuron in self.neurons.values():
            yield neuron

    def append( self, nrn ):
        """
            Add a new neuron to the NeuronList.
        """
        if type( nrn ) is NeuronObj:
            if nrn.id not in self.ids():
                self.neurons[nrn.id] = nrn
            else:
                print( 'Skipping {} ({}), already in list'.format(nrn.name, nrn.id) )
        else:
            raise TypeError('Only NeuronObjs can be added!')
        return self

    def loc( self, ind ):
        """
            Select only those neurons
        """
        try:
            out = self.neurons[ self.ids()[ind] ]
        except TypeError:
            raise TypeError
        except IndexError:
            raise IndexError
        return out

    def slice_by_id( self, id_list ):
        """
            Returns a new neuron list with only the members from set of ids.

            Parameters
            ----------
            id_list : list of ints or single int
                List of ids (or single id) of neurons to select.

            Returns
            -------
            NeuronList
                NeuronList with the same CatmaidInterface and other metadata, but only for those neurons specified.
        """
        if type(id_list) is int:
            id_list = [id_list]

        neurons_sampled = {id:self.neurons[id] for id in id_list if id in self.neurons.keys()}
        if not set(self.neurons.keys()).issuperset( set(id_list) ):
            print("Note: List of ids includes items not in the NeuronList!")
        return NeuronList( neurons_sampled,
                    CatmaidInterface = self.CatmaidInterface,
                    project_name = self.project_name,
                    export_date = self.export_date
                    )

    def slice_by_annotation( self, annos ):
        """
            Return a NeuronList that only contains neurons with the annotation(s) provided.

            Parameters
            ----------
            annos : list of strings (or single string)
                List of annotation strings to select.

            Returns
            -------
            NeuronList
                New neuron list with only those neurons with the specified annotations.
        """

        anno_ids = self.CatmaidInterface.parse_annotation_list( annos )
        neurons_sliced = {}
        for neuron in self:
            for anno in anno_ids:
                if anno in neuron.annotations:
                    neurons_sliced[ neuron.id ] = neuron
                    continue
        return NeuronList( neurons_sliced,
                    CatmaidInterface = self.CatmaidInterface,
                    project_name = self.project_name,
                    export_date = self.export_date
                    )    #

    @classmethod
    def from_id_list( cls, id_list, CatmaidInterface, with_tags = True, with_annotations = True, project_name = None, max_neurons_per_post=100):
        """
            Initialize a NeuronList object from a list of skeleton ids.

            Parameters
            ----------
            id_list : list of ints
                List of neuron ids to pull.

            CatmaidInterface : CatmaidDataInterface
                CatmaidDataInterface object for a Catmaid instance.

            Returns
            ----------
            NeuronList object
                List of NeuronObj for the specified ids, pulled from a catmaid server.

        """
        d = CatmaidInterface.get_skeleton_data(
                                    id_list,
                                    with_tags = with_tags,
                                    with_annotations = with_annotations,
                                    with_connectors = True,
                                    max_neurons_per_post = max_neurons_per_post
                                    )
        neurons = { id : NeuronObj.from_catmaid_data(id, d['skeletons'][str(id)], CatmaidInterface=CatmaidInterface) for id in id_list if id is not None}
        return cls( neurons, CatmaidInterface = CatmaidInterface, project_name = project_name, export_date = datetime.datetime.now().strftime('%c' ) )


    @classmethod
    def from_annotations( cls, annotation_list, CatmaidInterface, with_tags = True, with_annotations = True, project_name = None, max_neurons_per_post=100):
        """
            Initialize a NeuronList object from a list of annotations.

            Parameters
            ----------
            id_list : list of ints
                List of neuron ids to pull.

            CatmaidInterface : CatmaidDataInterface
                CatmaidDataInterface object for a Catmaid instance.

            Returns
            ----------
            NeuronList object
                Fancy list of NeuronObjs

        """
        id_list = CatmaidInterface.get_ids_from_annotations( annotation_list,
                                                             flatten=True )
        return cls.from_id_list(id_list,
                                CatmaidInterface,
                                with_tags=with_tags,
                                with_annotations=with_annotations,
                                project_name=project_name,
                                max_neurons_per_post=max_neurons_per_post )

    @classmethod
    def from_msc_json_import( cls, filename ):
        """
            Import files from json text data.

            Parameters
            ----------
            filename : str
                Filename of data

            Returns
            ----------
            NeuronList object
                Fancy list of NeuronObjs
        """
        with open( filename, 'r' ) as f:
            project_data = json.load( f )

        project_name = project_data['project_name']
        export_date = project_data['export_date']

        neurons_info_list = self._project_data_to_neuron_info( project_data )

        neurons = { neuron_info['id']: NeuronObj( neuron_info ) for neuron_info in neurons_info_list }
        return cls( neurons, export_date = export_date, project_name = project_name )

    def _project_data_to_neuron_info( self, project_data ):

        neuron_info_list = []
        for id_str in project_data['neuron_info']:
            neuron_data = project_data['neuron_info'][id_str]
            neuron_info = { 'id': int( id_str ),
                            'name': neuron_data['name'],
                            'tags': neuron_data['node_annotations']
                            }
            neuron_info['nodeids'] = []
            neuron_info['nodeloc'] = {}
            for row in neuron_data['morphology']['node_properties']:
                neuron_info['nodeids'].append(row[0])
                neuron_info['nodeloc'][ row[0] ] = row[1:]

            neuron_info['nodeparent'] = {}
            for row in neuron_data['topology']['edge_list']:
                neuron_info['nodeparent'][ row[0] ] = row[1]

            neuron_info['radius'] = {}
            if neuron_data['morphology']['node_property_list'] is not None:
                if 'radius' in neuron_data['morphology']['node_property_list']:
                    rad_ind = neuron_data['morphology']['node_property_list'].index('radius')
                    for row in neuron_data['morphology']['node_properties']:
                        if row[rad_ind] > 0:
                            neuron_info['radius'][ row[0] ] = row[rad_ind]
                        else:
                            neuron_info['radius'][ row[0] ] = None
                else:
                    for row in neuron_data['morphology']['node_properties']:
                        neuron_info['radius'][ row[0] ] = None
            else:
                for row in neuron_data['morphology']['node_properties']:
                    neuron_info['radius'][ row[0] ] = None

            conn_post_ids = [ row[1] for row in neuron_data['connectivity_post'] ]
            connector_data_post, connector_locs_post = self._dict_to_connector_data( conn_post_ids, project_data )

            conn_pre_ids = [ row[1] for row in neuron_data['connectivity_pre'] ]
            connector_data_pre, connector_locs_pre = self._dict_to_connector_data( conn_pre_ids, project_data )

            neuron_info['inputs'] = InputSynapseListObj( connector_data_post, connector_locs_post, neuron_info['id'] )
            neuron_info['outputs'] = OutputSynapseListObj( connector_data_pre, connector_locs_pre, neuron_info['id'] )

            neuron_info_list.append( neuron_info )

        return neuron_info_list

    def _dict_to_connector_data( self, connector_list, project_data ):
        connector_data = []
        connector_locs = {}

        for conn_id in connector_list:
            conn_dat = project_data[ 'connector_info' ][ str( conn_id ) ]
            connector_locs[conn_id] = conn_dat['node_properties'][1:]
            conn_dict = {}
            conn_dict['postsynaptic_to'] = [ row[1] for row in conn_dat['connectivity_post'] ]
            conn_dict['postsynaptic_to_node'] = [ row[0] for row in conn_dat['connectivity_post'] ]
            conn_dict['presynaptic_to'] = conn_dat['connectivity_pre'][1]
            conn_dict['presynaptic_to_node'] = conn_dat['connectivity_pre'][0]
            connector_data.append( [conn_id, conn_dict] )

        return connector_data, connector_locs

    def save_to_pickle( self, filename ):
        f = gzip.GzipFile(filename, 'wb')
        nrn_copy = copy.deepcopy(self)
        nrn_copy.CatmaidInterface = None    # Strip out user-specific catmaid interface details
        pickle.dump( self, f, protocol=2)
        f.close()
        return None

    @classmethod
    def load_from_pickle( cls, filename, catmaid_interface = None ):
        """
        Load a saved neuron list from a gzipped and pickled object.
        Parameters
        ----------
        filename : string
            Filename of the pickled object.

        Returns
        ----------
            NeuronList object
                New NeuronList. Useful because NeuronList definition is in active development.
        """
        f = gzip.GzipFile(filename, 'rb')
        nrn_list = pickle.load(f)
        f.close()
        return NeuronList( nrn_list.neurons,
                           CatmaidInterface = catmaid_interface,
                           project_name = nrn_list.project_name,
                           export_date = nrn_list.export_date
                           )

    def ids( self ):
        """
            Returns the skeleton ids of the neurons in the list.
        """
        return list( self.neurons.keys() )

    def names( self ):
        """
            Returns the names of the neurons in the list.
        """
        return [ self.neurons[nrn].name for nrn in self.neurons ]

    def append_catmaid_interface( self, CatmaidInterface ):
        """
            Add a CatmaidInterface object to a neuron object
        """
        self.CatmaidInterface = CatmaidInterface

    def annotations_from_neurons( self ):
        """
            Retrieves annotations for the neurons in the NeuronList.
            Parameters
            ----------
            None

            Returns
            ----------
            dict
                Dict of annotation entries

        """
        anno_dat = self.CatmaidInterface.get_annotations( )
        anno_dict = {}
        for anno_id in anno_dat['annotations']:
            anno_dict[ int(anno_id) ] = {'str' : anno_dat['annotations'][anno_id], 'skids': [] }
        for skid in anno_dat['skeletons']:
            for anno_info in anno_dat['skeletons'][skid]:
                anno_dict[ anno_info['id'] ][ 'skids' ].append( int(skid) )
        return anno_dict

    def get_adjacency_matrix( self, input_normalized = False ):
        """
            Build a weighted adjacency matrix from neurons

            Parameters
            ----------
            input_normalized : Boolean (default = False)
                Determine if synapses should be normalized by the total inputs on the postsynaptic neuron.

            Returns
            ----------
            A : numpy array
                Adjacency matrix with the number of synapses (or normalized synapses) from the column neuron onto the row neuron.

            skid_to_ind : dict
                Dictionary for mapping skeleton ids (keys) onto matrix indices (values)

            ind_to_skid : dict
                Dictionary for mapping matrix indices (keys) onto skeleton ids (values)
        """
        A = np.zeros( (len(self.neurons), len(self.neurons)) )
        ids = self.ids()
        skid_to_ind = { skid:ii for ii, skid in enumerate(self.neurons) }
        ind_to_skid = { ii:skid for ii, skid in enumerate(self.neurons) }
        for nrn in self.neurons:
            for conn_id in nrn.outputs.target_ids:
                for targ in nrn.outputs.target_ids[conn_id]:
                    if targ in ids:
                        if input_normalized is True:
                            A[ skid_to_ind[ targ ], skid_to_ind[ nrn.id ]] += 1.0 / self.neurons[ targ ].inputs.num()
                        else:
                            A[ skid_to_ind[ targ ], skid_to_ind[ nrn.id ]] += 1
        return A, skid_to_ind, ind_to_skid

    def export_adjacency_matrix( self, filename, index='id', input_normalized = False, delimiter=','):
        """
            Export an adjacecy matrix to a file
        """

        if index is not 'id' and index is not 'name':
            print('Index must be either ''id'' or ''name''')
            return -1

        with open( filename, 'w') as fid:
            A, skid_to_ind, ind_to_skid = self.get_adjacency_matrix( input_normalized = input_normalized )
            writer = csv.writer( fid, delimiter=delimiter )
            if index is 'id':
                writer.writerow( ['']+list(skid_to_ind.keys()) )
            else:
                names = self.names()
                writer.writerow( ['']+names)

            for i, row in enumerate(A):
                if index is 'id':
                    writer.writerow( [ ind_to_skid[i] ] + list(row) )
                else:
                    writer.writerow( [names[i]] + row.tolist() )
        return 1



    def group_adjacency_matrix( self, groups, func=np.sum, input_normalized = False ):
        """
            Adjacency matrix where the entries are for groups, not neurons.
            Groups come in a list of lists of skeleton ids.
        """
        A, skid_to_ind, ind_to_skid = self.get_adjacency_matrix( neurons, input_normalized = input_normalized )
        Agr = np.zeros( ( len(groups), len(groups) ) )
        for ii, grp_post in enumerate( groups ):
            for jj, grp_pre in enumerate( groups ):
                Ared = A[ [ skid_to_ind[ post ] for post in grp_post],:][:,[skid_to_ind[pre] for pre in grp_pre] ]
                Agr[ ii, jj ] = func( Ared )
        return Agr

    def find_ids_by_name( self, name_pattern ):
        """
            Use regex to find sk_ids of neurons that match a given search pattern.
        """
        return [nrn.id for nrn in self if re.search(name_pattern, nrn.name) is not None]

    def number_inputs( self ):
        """
            Return the number of inputs per neuron
        """
        return [nrn.inputs.num() for nrn in self]

    def number_outputs( self ):
        """
            Return the number of inputs per neuron
        """
        return [nrn.outputs.num() for nrn in self]

    def msc_json_export( self, filename, project_name=None, datestr=None, morpho_columns=None, topo_columns=None, annotation_id_whitelist = None ):
        if datestr is None:
            if self.CatmaidInterface is not None:
                datestr = self.CatmaidInterface.export_date
            else:
                datestr = datetime.datetime.now().strftime('%c' )

        if project_name is None:
            if self.CatmaidInterface is not None:
                project_name = self.CatmaidInterface.project_name

        neuron_info = self._msc_neuron_info( morpho_columns, topo_columns, annotation_id_whitelist )
        connector_info = self._msc_connector_info( )
        annotation_info = self._msc_annotation_info( annotation_id_whitelist )

        f_list = open( filename, 'w')
        json.dump( {'project_name':project_name, 'export_date':datestr, 'neuron_info':neuron_info, 'connector_info':connector_info, 'annotation_info':annotation_info}, f_list  )
        f_list.close()

    def _msc_connector_info( self ):
        connector_list = []
        connector_location_list = {}
        for neuron in self.neurons:
            connector_list = connector_list + neuron.inputs.conn_ids + neuron.outputs.conn_ids
            connector_location_list = {**connector_location_list,**neuron.inputs.locs,**neuron.outputs.locs}
        connector_list = list(set(connector_list))

        connector_info = {}

        conn_dat = self.CatmaidInterface.get_connector_data( connector_list, proj_opts )

        for conn in conn_dat:
            connector_info[ conn[0] ] = {}
            connector_info[ conn[0] ]['connector_annotation'] = None
            connector_info[ conn[0] ]['node_properties'] = [conn[0]] + connector_location_list[conn[0]]
            connector_info[ conn[0] ]['topology'] = None
            connector_info[ conn[0] ]['connectivity_pre'] = (conn[1]['presynaptic_to_node'], conn[1]['presynaptic_to'])
            connector_info[ conn[0] ]['connectivity_post'] =  list( zip(conn[1]['postsynaptic_to_node'], conn[1]['postsynaptic_to']) )
        return connector_info

    def _msc_neuron_info( self, morpho_columns=None, topo_columns=None, annotation_id_whitelist = None):
        neuron_info = {}
        for nrn in self.neurons:
            nid = nrn.id
            neuron_info[nid] = {}                        # Unique object id for the neuron.
            neuron = self.neurons[nid]
            neuron_info[nid]['name'] = neuron.name                 # Neuron name, string
            nrn_annos = self.annotations_from_neurons( {nid: neurons[nid]} )
            if annotation_id_whitelist is None:
                annotation_ids =  nrn_annos.keys()
            else:
                annotation_ids = [key for key in nrn_annos.keys() if key in annotation_id_whitelist]
            neuron_info[nid]['neuron_annotations'] = list( annotation_ids )   # A list of annotation ids

            # Morphology is everything related to nodes
            if morpho_columns is not None:                    # Morpho columns needs to be dict of dicts. First, property name, second node id.
                node_properties = [ [id]+neuron.nodeloc[id]+[morpho_columns[prop][id] for prop in morpho_columns ] for id in neuron.nodeloc ]
                node_property_list = [prop for prop in morpho_columns]
            else:
                node_properties = [ [id]+neuron.nodeloc[id] for id in neuron.nodeloc ]
                node_property_list = None
            neuron_info[nid]['morphology'] = {'node_properties': node_properties, 'node_property_list': node_property_list}

            # Topology is everything related to how nodes connect within a neuron
            topology = {}
            edge_list = [ [node_id, neuron.nodeparent[node_id]] for node_id in neuron.nodeparent ]
            edge_property_list = None

            # if topo_columns is not None:
            #     edge_list = [ [ node_id, neuron.nodeparent[ node_id ] ] + [ topo_columns[prop][ ( node_id, neuron.nodeparent[node_id] ) ] for prop in topo_columns] for node_id in neuron.nodeparent ]
            #     edge_property_list = [ prop for prop in topo_columns ]
            # else:
            #     edge_list = [ [node_id, neuron.nodeparent[node_id]] for node_id in neuron.nodeparent ] ]
            #     edge_property_list = None
            neuron_info[nid]['topology'] = {'edge_list': edge_list, 'edge_property_list': edge_property_list}

            # Connectivity is how synapses and other connections between neurons relate to anaotmy
            neuron_info[nid]['connectivity_pre'] = []
            for conn_id in neuron.outputs.from_node_ids:
                neuron_info[nid]['connectivity_pre'].append( ( neuron.outputs.from_node_ids[conn_id], conn_id) )
            neuron_info[nid]['connectivity_post'] = []
            for conn_id in neuron.inputs.target_node_ids:
                for node_id in neuron.inputs.target_node_ids[conn_id]:
                    neuron_info[nid]['connectivity_post'].append( (node_id, conn_id) )

            neuron_info[nid]['connectivity_undirected'] = None

            neuron_info[nid]['node_annotations'] = neuron.tags
        return neuron_info

    def _msc_annotation_info( self, annotation_id_whitelist = None ):
        anno_dat = self.annotations_from_neurons( )
        annotation_info = {}
        if annotation_id_whitelist is not None:
            anno_id_list = [anno for anno in anno_dat if anno in annotation_id_whitelist]
        else:
            anno_id_list = anno_dat.keys()
        for anno in anno_id_list:
            annotation_info[anno] = {'anno_string': anno_dat[anno]['str'], 'ids': anno_dat[anno]['skids'] }
        return annotation_info

class SynapseListObj:
    def __init__(self, locs, conndata ):
        self.conn_ids = [ dat[0] for dat in conndata ]
        self.locs = locs

class InputSynapseListObj(SynapseListObj):
    def __init__(self, conndata, locs, self_id):
        SynapseListObj.__init__(self, locs, conndata )
        self.target_node_ids = {}
        for dat in conndata:
            for skid, nid in zip( dat[1]['postsynaptic_to'], dat[1]['postsynaptic_to_node'] ):
                if skid == self_id:
                    if dat[0] in self.target_node_ids:
                        self.target_node_ids[ dat[0] ].append(nid)
                    else:
                        self.target_node_ids[ dat[0] ] = [nid]
        self.from_ids = { dat[0] : dat[1]['presynaptic_to'] for dat in conndata }
        self.from_node_ids = { dat[0] : dat[1]['presynaptic_to_node'] for dat in conndata }
    def num(self):
        return sum( map( lambda x: len(x), self.target_node_ids.values() ) )

class OutputSynapseListObj(SynapseListObj):
    def __init__(self, conndata, locs, self_id):
        SynapseListObj.__init__(self, locs, conndata )
        self.from_node_ids = {dat[0] : dat[1]['presynaptic_to_node'] for dat in conndata }
        self.target_ids = { dat[0] : dat[1]['postsynaptic_to'] for dat in conndata }
        self.target_node_ids = {dat[0] : dat[1]['postsynaptic_to_node'] for dat in conndata }

    def num_targets( self ):
        return {id : len(self.target_ids[id]) for id in self.target_ids}

    def num_targets_connector( self, conn_id):
        if conn_id in self.target_ids:
            return len( self.target_ids[conn_id])
        else:
            print( 'No such presynaptic connector id in neuron' )

    def num(self):
        return sum( self.num_targets().values() )

class SynapseObject:
    def __init__(self, conn_ids, proj_opts ):
        conndata = ci.get_connector_data( conn_ids, proj_opts )
        self.connectors = { dat[0] : dat[1] for dat in conndata }

class NeuronObj:
    def __init__(self, neuron_info_dict ):
        self.id = neuron_info_dict['id']
        self.name = neuron_info_dict['name']
        self.tags = neuron_info_dict['tags']

        self.nodeids = neuron_info_dict['nodeids']
        self.nodeloc = neuron_info_dict['nodeloc']
        self.node2ind = { nid: i for i, nid in enumerate( neuron_info_dict['nodeids'] ) }
        self.nodeparent = neuron_info_dict['nodeparent']
        self.radius = neuron_info_dict['radius']

        temp_root = [nid for nid in neuron_info_dict['nodeparent'] if neuron_info_dict['nodeparent'][nid] is None]
        self.root = temp_root[0]

        self.A = sparse.dok_matrix(
            ( len( neuron_info_dict['nodeloc'] ), len( neuron_info_dict['nodeloc'] ) ), dtype=np.float32 )
        self.Ab = sparse.dok_matrix(
            ( len( neuron_info_dict['nodeloc'] ), len( neuron_info_dict['nodeloc'] ) ), dtype=np.float32 )
        for key in neuron_info_dict['nodeparent'].keys():
            if neuron_info_dict['nodeparent'][key] is not None:
                self.A[
                    self.node2ind[ key ],
                    self.node2ind[ neuron_info_dict['nodeparent'][ key ] ]
                    ] = spatial.distance.euclidean( neuron_info_dict['nodeloc'][ key ], neuron_info_dict['nodeloc'][ neuron_info_dict['nodeparent'][ key ] ] )
                self.Ab[
                    self.node2ind[ key ],
                    self.node2ind[ neuron_info_dict['nodeparent'][ key ] ]
                    ] = 1

        self.inputs = neuron_info_dict['inputs']
        self.outputs = neuron_info_dict['outputs']
        self.annotations = neuron_info_dict['annotations']

    @classmethod
    def from_catmaid_data( cls, skid, skdata, CatmaidInterface ):
        """

        """
        neuron_info_dict = {}
        neuron_info_dict['id'] = skid

        neuron_info_dict['reviews'] = skdata[3]
        neuron_info_dict['annotations'] = skdata[4]
        neuron_info_dict['name'] = CatmaidInterface.get_neuron_name(skid)
        neuron_info_dict['tags'] = skdata[2]

        neuron_info_dict['nodeids'] = [nd[0] for nd in skdata[0]]
        neuron_info_dict['nodeloc'] = {nd[0]: nd[3:6] for nd in skdata[0]}
        neuron_info_dict['nodeparent'] = {nd[0]: nd[1] for nd in skdata[0]}
        neuron_info_dict['radius'] = {nd[0]: nd[6] for nd in skdata[0]}

        pre_conn_ids = [dat[1] for dat in skdata[1] if dat[2] == 0]
        post_conn_ids = [dat[1] for dat in skdata[1] if dat[2] == 1]

        pre_conn_locs = {conn_row[1]: conn_row[3:6] for conn_row in skdata[1] if conn_row[1] in pre_conn_ids}
        post_conn_locs = {conn_row[1]: conn_row[3:6] for conn_row in skdata[1] if conn_row[1] in post_conn_ids}

        neuron_info_dict['inputs'] = InputSynapseListObj( CatmaidInterface.get_connector_data( post_conn_ids ), post_conn_locs , skid )
        neuron_info_dict['outputs'] = OutputSynapseListObj( CatmaidInterface.get_connector_data( pre_conn_ids ), pre_conn_locs, skid )

        neuron_info_dict['annotations'] = CatmaidInterface.get_annotations_for_objects([skid])

        return cls( neuron_info_dict )

    @classmethod
    def from_catmaid(cls, skid, CatmaidInterface, with_tags = True, max_neurons_per_post=100):
        neuron_info_dict = {}
        neuron_info_dict['id'] = skid

        skdata = CatmaidInterface.get_skeleton_data( skid, with_tags=with_tags, max_neurons_per_post=max_neurons_per_post )
        neuron_info_dict['reviews'] = skdata[4]
        neuron_info_dict['annotations'] = skdata[5]
        neuron_info_dict['name'] = skdata[6]
        neuron_info_dict['tags'] = skdata[2]

        neuron_info_dict['nodeids'] = [nd[0] for nd in skdata[0]]
        neuron_info_dict['nodeloc'] = {nd[0]: nd[3:6] for nd in skdata[0]}
        neuron_info_dict['nodeparent'] = {nd[0]: nd[1] for nd in skdata[0]}
        neuron_info_dict['radius'] = {nd[0]: nd[6] for nd in skdata[0]}

        pre_conn_ids = [dat[1] for dat in skdata[1] if dat[2] == 0]
        post_conn_ids = [dat[1] for dat in skdata[1] if dat[2] == 1]

        pre_conn_locs = {conn_row[1]: conn_row[3:6] for conn_row in skdata[1] if conn_row[1] in pre_conn_ids}
        post_conn_locs = {conn_row[1]: conn_row[3:6] for conn_row in skdata[1] if conn_row[1] in post_conn_ids}

        neuron_info_dict['inputs'] = InputSynapseListObj( CatmaidInterface.get_connector_data( post_conn_ids ), post_conn_locs , skid )
        neuron_info_dict['outputs'] = OutputSynapseListObj( CatmaidInterface.get_connector_data( pre_conn_ids ), pre_conn_locs, skid )
        neuron_info_dict['annotations'] = CatmaidInterface.get_annotations_for_objects([skid])
        return cls( neuron_info_dict )

    def __str__( self ):
        return self.name

    def strahler_filter( self, min_sn ):
        """
            Create a new neuron where branches below a certain point are filtered out. Currently this removes synapse info.
        """
        sn = self.strahler_number()

        if min_sn < 0:
            min_sn = np.max( list(sn.values()) ) + min_sn

        inds_to_remove = [nid for nid in sn if sn[nid] < min_sn]
        nni = {}
        nni['id'] = self.id
        nni['name'] = self.name + ' (Strahler ≥ {})'.format(str(min_sn))

        new_tags = {tag : list( set(self.tags[tag]).difference(set(inds_to_remove))   ) for tag in self.tags}
        nni['tags'] = new_tags

        nni['nodeids'] = [nid for nid in self.nodeids if nid not in inds_to_remove]
        nni['nodeloc'] = {nid: self.nodeloc[nid] for nid in self.nodeloc if nid not in inds_to_remove}

        nni['nodeparent'] = {nid: self.nodeparent[nid] for nid in self.nodeparent if nid not in inds_to_remove}
        nni['radius'] = {nid: self.radius[nid] for nid in self.radius if nid not in inds_to_remove}

        input_obj = self.inputs
        input_obj.target_node_ids = {cid:list( set( self.inputs.target_node_ids[cid]).difference( set( inds_to_remove ) ) ) for cid in self.inputs.target_node_ids }
        nni['inputs'] = input_obj

        output_obj = self.outputs
        for cid in self.outputs.from_node_ids:
            if self.outputs.from_node_ids[cid] in inds_to_remove:
                output_obj.from_node_ids[cid] = np.NaN
        nni['outputs'] = output_obj

        nni['annotations'] = self.annotations
        #return NeuronObj( nni )
        return NeuronObj( nni )


    def cable_length(self):
        """
            Returns the total cable length of the neuron.
        """
        return self.A.sum()

    def node_count(self):
        """
            Returns the number of nodes in the skeleton
        """
        return len(self.nodeids)

    def find_end_nodes(self):
        """
            Returns a list of node ids that are end nodes (have no children)
        """
        y = np.where(self.Ab.sum(0) == 0)[1]
        return [self.nodeids[ind] for ind in y]

    def find_branch_points(self):
        """
            Returns a list of node ids that are branch points (have multiple children)
        """
        y = np.where(self.Ab.sum(0) > 1)[1]
        return [self.nodeids[ind] for ind in y]

    def minimal_paths(self):
        """
            Returns list of lists, the minimally overlapping paths from each end
            point toward root
        """
        D = self.dist_to_root(by_nodes=True)
        ids_end = self.find_end_nodes()

        ends_sorted = [ids_end[ind] for ind in np.argsort(
            D[[self.node2ind[id] for id in ids_end]])[::-1]]
        not_visited = [True] * len(self.nodeids)
        min_paths = []

        for start_nd in ends_sorted:
            nd = start_nd
            min_paths.append([nd])   # Start a new list with this end as a seed
            while not_visited[self.node2ind[nd]] and (self.nodeparent[nd] is not None):
                not_visited[self.node2ind[nd]] = False
                nd = self.nodeparent[nd]
                min_paths[-1].append(nd)
        min_paths_sorted = [ min_paths[ind] for ind in np.argsort( list( map( len, min_paths ) ) )[::-1] ]
        return min_paths_sorted

    def strahler_number( self ):
        """
            Computes strahler number for a neuron
        """
        paths = self.minimal_paths()[::-1]
        sn = {}
        for nid in self.nodeids:
            sn[nid] = 0

        for path in paths:
            sn[path[0]] = 1
            for ii, nid in enumerate(path[1:]):
                if sn[nid] == sn[path[ii]]:
                    sn[nid] = sn[path[ii]] + 1
                elif sn[nid] > sn[path[ii]]:
                    continue
                else:
                    sn[nid] = sn[path[ii]]
        return sn

    def split_into_components(self, nids, from_parent=True):
        """
            Return n-component list, each element is a list of node ids in the component.
            nids is a list of child nodes that will be split from their parent node.
            if from_parent is toggled false, parents divorce childen and not the
            default.
        """
        Ab_sp = copy.deepcopy(self.Ab)

        if from_parent:
            for id in nids:
                nind = self.node2ind[id]
                Ab_sp[:, nind] = 0
        else:
            for id in nids:
                nind = self.node2ind[id]
                Ab_sp[nind, :] = 0

        ncmp, cmp_label = csgraph.connected_components(Ab_sp, directed=False)

        cmps = list()
        for cmp_val in range(ncmp):
            comp_inds = np.where(cmp_label == cmp_val)
            cmps.append([self.nodeids[ind] for ind in comp_inds[0]])

        cmp_label_dict = {self.nodeids[ind]:cmp for ind,cmp in enumerate(cmp_label) }

        return cmps, cmp_label_dict

    def dist_to_root(self, by_nodes=False ):
        """
            Returns distance to root for each node in nrn as an array. If by_nodes is True, counts nodes not spatial distances.
        """
        if by_nodes:
            D = csgraph.shortest_path(
                self.Ab.transpose(), directed=True, unweighted=False, method='D')
        else:
            D = csgraph.shortest_path(
                self.A.transpose(), directed=True, unweighted=False, method='D')
        return D[self.node2ind[self.root]]

    def split_by_tag( self, tag_str ):
        """
            Returns components of a neuron split by a specific tag
        """
        nids = self.tags[ tag_str ]
        cmps, cmp_label = self.split_into_components( nids )
        return cmps, cmp_label

    def slabs( self ):
        """
            Returns slabs, the points between branch points, sorted by length.
        """
        nids = self.find_branch_points()
        cmps, cmp_label_dict = self.split_into_components( nids, from_parent=True )
        D = self.dist_to_root()
        cmps_ordered = []
        for cmp in cmps:
            cmps_ordered.append( sorted(cmp, key=lambda x:D[self.node2ind[x]]) )
        return sorted( cmps_ordered, key = len, reverse=True)

    def synaptic_partners( self, connection_type, min_synapses = 0, normalized=False ):
        """
            Get synaptic partners of the neuron and number of synapses.
            Parameters
            ----------
            connection_type : 'presynaptic' or 'postsynaptic'
                Whether inputs or outputs are computed.

            min_synapses : numeric (default 0)
                Minimum weight of synapses/fraction synapses to return

            normalized : Boolean (default False)
                Determines if synaptic weight is computed as number of synapses
                or fraction of all synapses in category.

            Returns
            -------
            partner_list : numpy array
                n x 2 array of partners, sorted descending by weight.
                Each row is the id, partner weight.
        """
        if connection_type == 'presynaptic':
            ids, syns = np.unique(list(self.inputs.from_ids.values()),return_counts=True)
        elif connection_type == 'postsynaptic':
            ids, syns = np.unique( [x for x in chain.from_iterable(self.outputs.target_ids.values()) ],return_counts=True )
        else:
            raise ValueError('connection_type must be \'presynaptic\' or \'postsynaptic\'')

        if normalized:
            syns = syns / np.sum(syns)

        # Reorder to start with strongest partners
        ids = ids[np.argsort(-syns)]
        syns = -np.sort(-syns)
        
        return np.vstack( (ids[syns>=min_synapses], syns[syns>=min_synapses]) ).T

def synaptic_partner_tables( neurons,
                       include_presynaptic=True,
                       include_postsynaptic=True,
                       use_name = False):
    """
        Return tables equivalent to the CATMAID connectivity widget.
        Parameters
        ----------

        Returns
        ----------
    """
    if include_presynaptic:
        input_table = []
        for nrn in neurons:
            input_dict = defaultdict(int)
            for presyn_id in nrn.inputs.from_ids.values():
                if presyn_id is not None:
                    input_dict[presyn_id] += 1
            if use_name:
                series_name = nrn.name
            else:
                series_name = nrn.id
            input_table.append( pd.Series(input_dict, name=series_name) )
        input_df = pd.DataFrame( input_table ).transpose().fillna(0)
    else:
        if use_name:
            series_names = [nrn.name for nrn in neurons]
        else:
            series_names = [nrn.id for nrn in neurons]
        input_df = pd.DataFrame( columns=series_names)

    if include_postsynaptic:
        output_table = []
        for nrn in neurons:
            output_dict = defaultdict(int)
            for output in nrn.outputs.target_ids:
                if nrn.outputs.target_ids[output] is not None:
                    for targ_id in nrn.outputs.target_ids[output]:
                        output_dict[targ_id] += 1
            if use_name:
                series_name = nrn.name
            else:
                series_name = nrn.id
            output_table.append( pd.Series(output_dict, name=series_name))

        output_df = pd.DataFrame( output_table ).transpose().fillna(0)
    else:
        if use_name:
            series_names = [nrn.name for nrn in neurons]
        else:
            series_names = [nrn.id for nrn in neurons]
        output_df = pd.DataFrame( columns=series_names)

    return input_df, output_df
