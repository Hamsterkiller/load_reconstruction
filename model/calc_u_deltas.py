import numpy
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import networkx as nx

from .power_system import PowerSystem
from .helpers import convert_to_relative_units


def calc_u_deltas(graph: nx.Graph):
    """
    Calculate voltage angles between node_from and node_to for each line:
    Pf = Gf*Vf + g * (Vf**2 - Vf*Vt*cos*(dd) - b*Vf*Vt*sin(dd)
    Pt= Gt*Vt + g * (Vt**2 - Vf*Vt*cos*(dd) + b*Vf*Vt*sin(dd)
    :param graph - networkx graph of the system
    :return:
    """

    # find initial list of edges with at least one unknown u value
    unknown_dd_edges = []
    for nf, nt, v in graph.edges(data=True):
        if graph.edges[nf, nt].get('dd', 1e6) > 1e5:
            unknown_dd_edges.append((nf, nt, v))

    # for each line in unknown_dd_edges
    angles = []
    for nf, nt, v in unknown_dd_edges:

        # get line parameters
        uf = graph.nodes[v['node_from']].get('u')
        ut = graph.nodes[v['node_to']].get('u')
        type = v['type']
        pf = v['p_from']
        pt = v['p_to']
        ktr = v['ktr']
        r = v['r']
        x = v['x']
        gsh = v['gsh']
        bsh = v['bsh']

        # calc other params
        ts = ktr
        if type != 1:
            ts = 1
        ut = ut / ts

        # shunt conductance
        G = gsh
        if type == 1:
            # for transformators equivalent scheme is G-like
            G_f = G * 1e-6
            G_t = 0.0
        else:
            # for ordinary lines equivalent scheme P-like
            G_f = (G / 2) * 1e-6
            G_t = (G / 2) * 1e-6

        # conductance
        R = r
        X = x

        # fix near-zero impedance values
        # if np.abs(R + 1j * X) < 1e-6:
        #     X = 0.04 / 100

        g = R / (R ** 2 + X ** 2)
        b = -X / (R ** 2 + X ** 2)

        if b == 0:
            val = (g * (uf**2 + ut**2) - (pf - pt) + G_f * uf**2 + G_t * ut**2) / (2 * g * uf * ut)
        else:
            val = (pf - pt - G_f * uf**2 + G_t * ut**2 - g * (uf**2 - ut**2)) / (2 * b * uf * ut)

        if val > 1.0:
            val = 1.0
        if val < -1.0:
            val = -1.0

        dd = 1e-5
        if b == 0:
            dd = np.arccos(val)
        else:
            dd = np.arcsin(val)

        angles.append(dd)

        graph.edges[nf, nt]['dd'] = dd

