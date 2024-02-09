import numpy
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import networkx as nx

from .power_system import PowerSystem
from .helpers import convert_to_relative_units


def calc_u_deltas(g: nx.Graph):
    """
    Calculate voltage angles between node_from and node_to for each line:
    Pf = Gf*Vf + g * (Vf**2 - Vf*Vt*cos*(dd) - b*Vf*Vt*sin(dd)
    Pt= Gt*Vt + g * (Vt**2 - Vf*Vt*cos*(dd) + b*Vf*Vt*sin(dd)
    :param ps - instance of the PowerSystem class
    :param node_u - U values for each node with known voltage
    :param line_flows - known active capacity flows p_from, p_to for each line
    :return:
    """

    # find initial list of edges with at least one unknown u value
    unknown_dd_edges = []
    for nf, nt, v in g.edges(data=True):
        if g.edges[nf, nt].get('dd', -1) < 0:
            unknown_dd_edges.append((nf, nt, v))

    # for each line in unknown_dd_edges
    for nf, nt, vv in unknown_dd_edges:
        pass

    print("Test!")

    # while unknown_u_edges:
    #     for nf, nt, v in unknown_u_edges:
    #         # print(f"{nf} - {nt}")
    #         if (g.nodes[v['node_from']].get('u', -1) > 0) & (g.nodes[v['node_to']].get('u', -1) > 0):
    #             unknown_u_edges.remove((nf, nt, v))
    #         if (g.nodes[v['node_from']].get('u', -1) < 0) & (g.nodes[v['node_to']].get('u', -1) > 0):
    #             solution = solve_flow_equations(g.nodes[v['node_from']].get('u', -1),
    #                                             g.nodes[v['node_to']].get('u', -1),
    #                                             g.edges[nf, nt].get('type'),
    #                                             g.edges[nf, nt].get('p_from'),
    #                                             g.edges[nf, nt].get('p_to'),
    #                                             g.edges[nf, nt].get('ktr'),
    #                                             g.edges[nf, nt].get('r'),
    #                                             g.edges[nf, nt].get('x'),
    #                                             g.edges[nf, nt].get('gsh'),
    #                                             g.edges[nf, nt].get('bsh'))
    #             g.nodes[v['node_from']]['u'] = solution['u']
    #             print(
    #                 f"Unknown voltage module = {solution['u']}, angle = {solution['dd']}, type = {v['type']}, ktr = {v['ktr']}, ut = {g.nodes[v['node_to']]['u']}")
    #             unknown_u_edges.remove((nf, nt, v))
    #         if (g.nodes[v['node_from']].get('u', -1) > 0) & (g.nodes[v['node_to']].get('u', -1) < 0):
    #             solution = solve_flow_equations(g.nodes[v['node_from']].get('u', -1),
    #                                             g.nodes[v['node_to']].get('u', -1),
    #                                             g.edges[nf, nt].get('type'),
    #                                             g.edges[nf, nt].get('p_from'),
    #                                             g.edges[nf, nt].get('p_to'),
    #                                             g.edges[nf, nt].get('ktr'),
    #                                             g.edges[nf, nt].get('r'),
    #                                             g.edges[nf, nt].get('x'),
    #                                             g.edges[nf, nt].get('gsh'),
    #                                             g.edges[nf, nt].get('bsh'))
    #             g.nodes[v['node_to']]['u'] = solution['u']
    #             print(
    #                 f"Unknown voltage module = {solution['u']}, angle = {solution['dd']}, type = {v['type']}, ktr = {v['ktr']}, uf = {g.nodes[v['node_from']]['u']}")
    #             unknown_u_edges.remove((nf, nt, v))
    #
    # print("Finished!")


# def calculate_delta():
#     # flow_from and flow_to vectors
#     line_n = line_flows.shape[0]
#     P_f = np.array(line_flows.flow).reshape(line_n, 1)
#     P_t = np.array(line_flows.flow_to).reshape(line_n, 1)
#
#     # bus voltages
#     voltages_f = line_flows[['node_from', 'node_to']].merge(right=node_u,
#                                                             left_on=['node_from'],
#                                                             right_on=['node'],
#                                                             how='left')
#     voltages_t = line_flows[['node_from', 'node_to']].merge(right=node_u,
#                                                             left_on=['node_to'],
#                                                             right_on=['node'],
#                                                             how='left')
#     u_f = np.array(voltages_f.u).reshape(line_n, 1)
#     u_t = np.array(voltages_t.u).reshape(line_n, 1)
#     unom_f = np.array(voltages_f.unom).reshape(line_n, 1)
#     unom_t = np.array(voltages_t.unom).reshape(line_n, 1)
#
#     # convert to relative values
#     # u_f = convert_to_relative_units(u_f, 'kV', unom_f)
#     # u_t = convert_to_relative_units(u_t, 'kV', unom_t)
#
#     # vetv topology parameter
#     ktr = np.array(line_flows.ktr).reshape(line_n, 1)
#     ts = np.ones(line_n).reshape(line_n, 1)
#     is_trans = line_flows.type == 1
#     ts[is_trans] = ktr[is_trans]
#     u_t = u_t / ts
#
#     # shunt conductance
#     # for ordinary lines equivalent scheme P-like
#     G = np.array(line_flows.g).reshape(line_n, 1)
#     G_f = G / 2 * 1e-6
#     G_t = G / 2 * 1e-6
#
#     # for transformators equivalent scheme is G-like
#     G_f[is_trans] = G[is_trans] * 1e-6
#     G_t[is_trans] = 0.0
#
#     # conductance
#     R = np.array(line_flows.r).reshape(line_n, 1)
#     X = np.array(line_flows.x).reshape(line_n, 1)
#     # idx = line_flows.index[(np.abs(line_flows.r) < 1e-6) & (np.abs(line_flows.x) < 1e-6)]
#     # X[idx] = 0.1
#     #X = np.maximum(np.array(line_flows.x).reshape(line_n, 1), 0.1)
#     g = R / (R**2 + X**2)
#     b = -X / (R**2 + X**2)
#
#     # calculate angle deltas for each vetv with all parameters known
#     # for lines with non-zero susceptance
#     non_zero_b = np.where(np.abs(b) >= 1e-6)[0]
#     zero_b = np.where(np.abs(b) < 1e-6)[0]
#     val = np.zeros(line_n).reshape(line_n, 1)
#
#     # for lines with non-zero b value: P_f - P_t
#     val[non_zero_b] = (P_f[non_zero_b] - P_t[non_zero_b] - G_f[non_zero_b] * u_f[non_zero_b]**2
#                        + G_t[non_zero_b] * u_t[non_zero_b]**2 - g[non_zero_b]
#                        * (u_f[non_zero_b]**2 - u_t[non_zero_b]**2)) / (2 * b[non_zero_b]
#                                                                        * u_f[non_zero_b] * u_t[non_zero_b])
#
#     # for lines with zero b value: P_f + P_t
#     val[zero_b] = (g[zero_b] * (u_f[zero_b]**2 + u_t[zero_b]**2) - (P_f[zero_b] - P_t[zero_b])
#                    + G_f[zero_b] * u_f[zero_b]**2 + G_t[zero_b] * u_t[zero_b]**2) \
#                   / (2 * g[zero_b] * u_f[zero_b] * u_t[zero_b])
#
#     invalid_idx_pos = np.where(val > 1.0)[0]
#     if invalid_idx_pos.shape[0] > 0:
#         val[invalid_idx_pos] = 1.0
#
#     invalid_idx_neg = np.where(val < -1.0)[0]
#     if invalid_idx_neg.shape[0] > 0:
#         val[invalid_idx_neg] = -1.0
#
#     dd = np.zeros(line_n)
#     dd[non_zero_b] = np.arcsin(val[non_zero_b]).reshape(-1)
#     dd[zero_b] = np.arccos(val[zero_b]).reshape(-1)
#
#     result = pd.concat([line_flows[['node_from', 'node_to', 'pnum']], pd.Series(dd).to_frame()], axis=1)
#     result.columns = ['node_from', 'node_to', 'pnum', 'u_angle']
#
#     return result


