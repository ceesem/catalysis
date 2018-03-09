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
from itertools import chain, cycle

# def plot_neuron_ipyvolume(nrn, with_synapses = False, presynaptic_color = 'crimson', postsynaptic_color = 'darkturquoise', color='black', **kwargs):
#     paths = nrn.minimal_paths()
#     ind_lookup = {nid:ind for ind, nid in enumerate(nrn.nodeloc)}
#     xyz = np.zeros((len(nrn.nodeloc),3))
#     for nid in nrn.nodeloc:
#         xyz[ind_lookup[nid],:] = nrn.nodeloc[nid]
#
#     nid_lines = []
#     for path in paths:
#         nid_path = [ [nid,path[ind+1]] for ind, nid in enumerate(path[:-1]) ]
#         nid_lines.append( list(chain.from_iterable(nid_path)) )
#     nid_line = list(chain.from_iterable(nid_lines))
#     lines = [ ind_lookup[nid] for nid in nid_line ]
#
#     p3.plot_trisurf(xyz[:,0],xyz[:,1],xyz[:,2],triangles=None,lines=lines, color=color)
#     return p3.current.container

def plot_neurons( nrns, with_synapses = False, color = (0.1,0.1,0.1), width=2, presynaptic_color = (0.9,0.1,0.1), postsynaptic_color = (0.1,0.4,0.9), layout=None ):
    """
        Plot neurons in an iterable collection of neuron using Plotly.
    """

    data = []
    if type(nrns) is cat.neurons.NeuronObj:
        nrns = [nrns]

    if type(with_synapses) is bool:
        with_synapses = {nrn.id : with_synapses for nrn in nrns}

    if type(color) is str:
        color_cycle = itertools.cycle( cl.get_cmap(color) )
        color = {nid.id : next(color_cycle) for nrn in nrns}
    elif len(np.shape( color )) == 1:
        color = {nrn.id : color for nrn in nrns}

    for nrn in nrns:

        paths = nrn.minimal_paths()
        xyz = []
        for path in paths:
            partial_xyz = [ nrn.nodeloc[nid] for nid in path ]
            partial_xyz.append( [np.nan] * 3)
            xyz.append(partial_xyz)
        xyz = np.array(list( chain.from_iterable(xyz) ))

        line = dict(width = width,
                    color = cl.to_hex( color[nrn.id] ) )

        data.append( go.Scatter3d(
            x = xyz[:,0],
            y = xyz[:,1],
            z = xyz[:,2],
            line = line,
            mode='lines',
            name = nrn.name ) )

        if 'soma' in nrn.tags:
            xyz0 = nrn.nodeloc[ nrn.tags['soma'][0] ]
            data.append( sphere_data(xyz0, color=color[nrn.id] ) )

        if with_synapses[nrn.id]:
            post_xyz = np.array([val for val in nrn.inputs.locs.values()])
            data.append( go.Scatter3d(
                x = post_xyz[:,0],
                y = post_xyz[:,1],
                z = post_xyz[:,2],
                showlegend = False,
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=2,
                    color=cl.to_hex(postsynaptic_color) )
            ) )

            pre_xyz = np.array([val for val in nrn.outputs.locs.values()])
            data.append( go.Scatter3d(
                x = pre_xyz[:,0],
                y = pre_xyz[:,1],
                z = pre_xyz[:,2],
                showlegend = False,
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=2,
                    color=cl.to_hex(presynaptic_color) )
            ) )

    layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    ) )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
    return fig

def sphere_data( xyz0, color, radius=2000, npts=10 ):
    """
        Generate the Mesh3d data for a sphere of a specific color, radius, and locatoin.
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


def slab_data( xyz, fig=None, color = (20,20,20), line=None ):
    """
        Generate Plotly Scatter3 data for a numpy array of xyz points as a sequential line.
    """

    if line is not None:
        line = dict(
        width = 2,
        color = cl.to_hex(color)
        )

    return go.Scatter3d(
        x = xyz[:,0],
        y = xyz[:,1],
        z = xyz[:,2],
        line = line,
        mode='lines'
        )
