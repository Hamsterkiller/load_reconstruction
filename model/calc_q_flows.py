import networkx as nx
import numpy
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from .power_system import PowerSystem
from .helpers import convert_to_relative_units


def calc_q_flows(graph: nx.Graph):
    """
    Calculate node reactive loads.
    @param graph networkx graph
    :return:
    """

    result_dict = []
    for nf, nt, v in graph.edges(data=True):

        if (v['node_from'] == 804130) & (v['node_to'] == 804530):
            a = 1

        # get line parameters
        uf = graph.nodes[v['node_from']].get('u')
        ut = graph.nodes[v['node_to']].get('u')
        type = v['type']
        if type == 2:
            result_dict.append({'node_from': v['node_from'], 'node_to': v['node_to'], 'q_from': 0, 'q_to': 0})
        else:
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
            B = bsh
            if type == 1:
                # for transformators equivalent scheme is G-like
                G_f = G * 1e-6
                G_t = 0.0
                B_f = B * 1e-6
                B_t = 0.0
            else:
                # for ordinary lines equivalent scheme P-like
                G_f = (G / 2) * 1e-6
                G_t = (G / 2) * 1e-6
                B_f = (B / 2) * 1e-6
                B_t = (B / 2) * 1e-6

            # conductance
            R = r
            X = x

            # fix near-zero impedance values
            # if np.abs(R + 1j * X) <= 1e-5:
            #     X = 0.04 / 100

            g = R / (R ** 2 + X ** 2)
            b = -X / (R ** 2 + X ** 2)

            dd = v['dd']

            # reactive power flows
            qf = b * (uf ** 2 - uf * ut * np.cos(dd)) - g * (uf * ut * np.sin(dd)) + B_f * uf ** 2
            qt = b * (ut ** 2 - uf * ut * np.cos(dd)) + g * (uf * ut * np.sin(dd)) + B_t * ut ** 2

            if (np.abs(qf) > 1000) | (np.abs(qt) > 1000):
                R = 1000
                g = R / (R ** 2 + X ** 2)
                b = -X / (R ** 2 + X ** 2)
                qf = b * (uf ** 2 - uf * ut * np.cos(dd)) - g * (uf * ut * np.sin(dd)) + B_f * uf ** 2
                qt = b * (ut ** 2 - uf * ut * np.cos(dd)) + g * (uf * ut * np.sin(dd)) + B_t * ut ** 2

            result_dict.append({'node_from': v['node_from'], 'node_to': v['node_to'], 'q_from': qf, 'q_to': qt})

    result_df = pd.DataFrame.from_records(result_dict)

    return result_df


