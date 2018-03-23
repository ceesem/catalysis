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
                    color=cl.to_hex(postsynaptic_color) )
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


def path_data( xyz, color = (20,20,20), line=None, width=2 ):
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
        mode='lines'
        )

def scatter_block( xyz, marker=None, color = None ):
    """

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