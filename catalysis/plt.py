import catalysis as cat
import matplotlib.pyplot as plt
import matplotlib.colors as cl

# import ipyvolume as ipv
# import ipyvolume.pylab as p3

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

import numpy as np
import scipy as sp
import re
from itertools import chain, cycle

def neuronlist_data( nrns,
                     with_synapses = False,
                     color = 'Set1',
                     width=2,
                     presynaptic_color = (0.9,0.1,0.1),
                     postsynaptic_color = (0.1,0.4,0.9) ):
    """
        Generates the graphics objects for a neuron list (or single neuron)
        without plotting.
    """
    if type(nrns) is cat.neurons.NeuronObj:
        nrns = [nrns]

    if type(with_synapses) is bool:
        with_synapses = {nrn.id : with_synapses for nrn in nrns}

    if type(color) is str:
        color_cycle = cycle( plt.get_cmap(color).colors )
        color = {nrn.id : next(color_cycle) for nrn in nrns}
    elif len(np.shape( color )) == 1:
        color = {nrn.id : color for nrn in nrns}

    data = []

    for nrn in nrns:
        data.append( neuronal_morphology_data( nrn, color=color[nrn.id], width=width ) )

        if 'soma' in nrn.tags:
            xyz0 = nrn.nodeloc[ nrn.tags['soma'][0] ]
            data.append( sphere_data(xyz0, color=color[nrn.id] ) )

        if with_synapses[nrn.id]:
            data.append(neuronal_synapse_data(nrn,synapse_type='post',color=postsynaptic_color, markersize=2) )
            data.append(neuronal_synapse_data(nrn,synapse_type='pre',color=presynaptic_color, markersize=2) )

    return data

def plot_neurons( nrns, with_synapses = False, color = 'Set1', width=2, presynaptic_color = (0.9,0.1,0.1), postsynaptic_color = (0.1,0.4,0.9), layout=None ):
    """
        Plot neurons in an iterable collection of neuron using Plotly.
    """


    if layout is None:
        layout = go.Layout( autosize=False,
                            width=1000,
                            height=1000,
                            margin=dict(
                                l=65,
                                r=50,
                                b=65,
                                t=90 )
                            )

    data = neuronlist_data( nrns,
                            with_synapses=with_synapses,
                            color=color,
                            width=width,
                            presynaptic_color=presynaptic_color,
                            postsynaptic_color=postsynaptic_color )

    fig = go.Figure( data=data, layout=layout )
    py.iplot(fig)
    return fig

def neuronal_morphology_data( nrn, color=(0,0,0), width=2 ):
    """
        Returns a Scatter3d graphics object for a neuronal morphology.
    """
    paths = nrn.minimal_paths()
    xyz = []
    for path in paths:
        partial_xyz = [ nrn.nodeloc[nid] for nid in path ]
        partial_xyz.append( [np.nan] * 3)
        xyz.append(partial_xyz)
    xyz = np.array( list( chain.from_iterable(xyz) ) )

    line = dict(width = width,
                color = cl.to_hex( color ) )

    return go.Scatter3d(
        x = xyz[:,0],
        y = xyz[:,1],
        z = xyz[:,2],
        line = line,
        mode='lines',
        name = nrn.name )

def neuronal_synapse_data( nrn, synapse_type, color, markersize ):
    """
        Generate a Scatter3d object for a neuron's synapses (type='pre', 'post', or 'all') with specified color and marker size.
    """

    if synapse_type is 'post':
        xyz = np.array([val for val in nrn.inputs.locs.values()])
    elif synapse_type is 'pre':
        xyz = np.array([val for val in nrn.outputs.locs.values()])
    elif synapse_type is 'all':
        xyz_in = np.array([val for val in nrn.inputs.locs.values()])
        xyz_out = np.array([val for val in nrn.outputs.locs.values()])
        xyz = np.stack( (xyz_in,xyz_out) )

    return go.Scatter3d(
                x = xyz[:,0],
                y = xyz[:,1],
                z = xyz[:,2],
                showlegend = False,
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=markersize,
                    color=cl.to_hex(color) )
            )



def sphere_data( xyz0, color, radius=2000, npts=10 ):
    """
        Generate the Mesh3d data for a sphere of a specific color, radius, and center.
    """
    phis = np.linspace(-np.pi/2, np.pi/2, npts)
    thetas = np.linspace(0, 2*np.pi , npts)
    pp,tt = np.meshgrid(phis,thetas)
    pp = pp.flatten()
    tt = tt.flatten()

    xs = radius * np.cos(pp)*np.cos(tt) + xyz0[0]
    ys = radius * np.cos(pp)*np.sin(tt) + xyz0[1]
    zs = radius * np.sin(pp) + xyz0[2]

    points2D = np.vstack([pp,tt]).T
    tri = sp.spatial.Delaunay(points2D)
    simplices = tri.simplices

    fig_prop = ff.create_trisurf(x=xs,y=ys,z=zs,simplices=simplices,colormap = [cl.to_hex(color),cl.to_hex(color)])
    return fig_prop['data'][0]


def path_data( xyz, color = (0.3,0.3,0.3), line=None, width=2, name=None ):
    """
        Generate Plotly Scatter3d line for a numpy array of xyz points treated as a sequence.
    """
    if type(xyz) is not np.ndarray:
        xyz = np.array(xyz)

    if line is None:
        line = dict( width = width, color = cl.to_hex(color) )

    return go.Scatter3d(
        x = xyz[:,0],
        y = xyz[:,1],
        z = xyz[:,2],
        line = line,
        mode='lines',
        name=name
        )

def scatter_block( xyz, marker=None, color = None ):
    """
        Given an Nx3 array of points, scatterplot them.
        

    """

    if type(xyz) is not np.ndarray:
        xyz = np.array(xyz)

    if marker is None:
        marker = dict( symbol='circle', size=2 )
    if color is not None:
        marker['color'] = cl.to_hex(color)

    return go.Scatter3d(
        x = xyz[:,0],
        y = xyz[:,1],
        z = xyz[:,2],
        marker=marker,
        mode='markers'
        )


def match_report_plot_data( nrn_q, nrns_t, matches, color_q=(0.2,0.2,0.2), color_match='Set1'):
    """
    Plot all neurons that were potential matches
    """
    nrn_q_data = neuronlist_data( nrn_q,
                                  with_synapses=False,
                                  color = color_q,
                                  width = 4 )
    match_nrns = []
    for row in matches[nrn_q.id].iterrows():
        rowname = row[0]
        try:
            match_id = int(re.search('\((\d+)\)$',rowname).groups()[0])
            match_nrns.append( nrns_t[match_id] )
        except:
            print( 'Warning! No matches!' )
            continue
    nrn_t_data = neuronlist_data( match_nrns,
                                  with_synapses=False,
                                  color=color_match,
                                  width=2 )
    return nrn_q_data + nrn_t_data

def connectivity_line_plot( A, plot_groups, colors, params=None ):
    """
    Produce a mesh-style line plot of neurons for Plotly.
    Usage:
        To view interactively:
            data, layout = connectivity_line_plot( A, plot_groups, colors )
            py.iplot( go.Figure(data=data, layout=layout) )
        To save to an svg file:
            data, layout = connectivity_line_plot( A, plot_groups, colors )
            py.plot( go.Figure(data=data, layout=layout), image='svg', image_filename='my_filename' ) 

    Parameters
    ----------
        A : NetworkX graph
            Directed matrix such that A['pre_id']['post_id'] = # synapses from pre to post.

        plot_groups : dict
            Dictionary such that plot_groups[group_name] = [ids in group]

        colors : dict
            Dictionary such that colors[ group_name ] = color for plotting.

        params : dict (optional, default None)
            Dictionary containing visualization parameters (spacing, etc)

        highlight_reciprocal : Boolean (optional, default False)
            Indicates whether edges part of a reciprocal connection are highlighted in the plot by a dashed line. (Not yet implemented)


    Returns
    -------
        Data Object for Plotly
            Contains all nodes and lines

        Layout Object for Plotly
            Spaces the layout for the intended visual parameters.

    """

    if params is None:
        params = dict()
    if 'gapwidth' not in params:
        params['gapwidth']=3            # Sets gap between groups, in units of number of nodes
    if 'height' not in params:
        params['height']=100            # Sets overall height
    if 'columnwidth' not in params:
        params['columnwidth']=10        # Width of the pre-to-post column units of number of nodes
    if 'nodesize' not in params:
        params['nodesize'] = 10         # Sets node size
    if 'linescaling' not in params:
        params['linescaling'] = 1       # Sets scaling relationship between adjacency matrix and line width
    if 'linealpha' not in params:
        params['linealpha'] = 0.4       # Sets opacity of lines
    if 'xgap' not in params:
        params['xgap'] = 3              # Sets distance between adjacent columns

    nclass = len(plot_groups)
    ngaps = nclass-1
    nnodes = len(A) + ngaps * params['gapwidth']
    deltay = params['height'] / nnodes

    # Compute node locations
    xvals = []
    gap_num = 0
    xvals.append(0)
    last_xval = 0
    while gap_num < len(plot_groups)-1:
        xvals.append( last_xval + params['columnwidth'] * deltay )
        xvals.append( last_xval + params['columnwidth'] * deltay + params['xgap'] )
        last_xval = xvals[-1]
        gap_num += 1
    xvals.append(last_xval + params['columnwidth'] * deltay  )

    xvals = np.array(xvals)
    yvals = np.arange(params['height'], 0, -1*deltay)

    yindex = {}
    index_num=0
    for group in plot_groups:
        for nid in plot_groups[group]:
            yindex[ nid ] = index_num
            index_num += 1
        index_num += params['gapwidth']-1

    # Make Scatter marker object for all nodes
    all_nodes = []
    for group in plot_groups:
        ydat = yvals[ [yindex[nid] for nid in plot_groups[group]] ] 
        for xpos in xvals:
            all_nodes.append(
                go.Scatter( x=xpos*np.ones(np.shape(plot_groups[group])),
                            y=ydat,
                            mode='markers',
                            showlegend=False,
                            marker=dict(
                                size = params['nodesize'],
                                color = 'rgba{}'.format( cl.to_rgba(colors[group]) ),
                                line = dict(
                                    width = 2,
                                    color = 'rgb(1,1,1)'
                                    )
                                )
                            )
                )

    # Make Scatter line object for all edges.
    all_lines = []
    for ii, group in enumerate(plot_groups):
        for nid in plot_groups[group]:
            for targ_id in A[nid]:
                all_lines.append( go.Scatter( x=[xvals[2*ii], xvals[2*ii+1]],
                                        y=[yvals[yindex[nid]], yvals[ yindex[targ_id]] ],
                                        mode='lines',
                                        showlegend=False,
                                        line=dict(
                                            width= A[nid][targ_id]['weight'] * params['linescaling'],
                                            color= 'rgba{}'.format( cl.to_rgba(colors[group],params['linealpha']))
                                            )
                                        )
                                )

    # Set a lightweight Layout with appropriate values for the parameters
    layout = go.Layout( 
                        height = 1.1*max(yvals),
                        width = 1.1*max(xvals),
                        xaxis = dict( visible=False ),
                        yaxis = dict( visible=False )
                        )

    return all_lines+all_nodes, layout

