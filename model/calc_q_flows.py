import numpy
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from .power_system import PowerSystem
from .helpers import convert_to_relative_units


def calc_q_flows(ps: PowerSystem, node_u: pd.DataFrame, line_flows: pd.DataFrame, u_deltas: np.ndarray) \
        -> list[np.ndarray, np.ndarray]:
    """
    Calculate node reactive loads.
    :param ps - instance of the PowerSystem class
    :param node_u - node voltage modules
    :param line_flows - line flows
    :param u_deltas - voltage angles
    :return:
    """

    # flow_from and flow_to vectors
    line_n = line_flows.shape[0]

    # bus voltages
    voltages_f = line_flows[['node_from', 'node_to']].merge(right=node_u,
                                                            left_on=['node_from'],
                                                            right_on=['node'],
                                                            how='left')
    voltages_t = line_flows[['node_from', 'node_to']].merge(right=node_u,
                                                            left_on=['node_to'],
                                                            right_on=['node'],
                                                            how='left')
    u_f = np.array(voltages_f.u).reshape(line_n, 1)
    u_t = np.array(voltages_t.u).reshape(line_n, 1)

    unom_f = np.array(voltages_f.unom).reshape(line_n, 1)
    unom_t = np.array(voltages_t.unom).reshape(line_n, 1)

    # convert to relative values
    # u_f = convert_to_relative_units(u_f, 'kV', unom_f)
    # u_t = convert_to_relative_units(u_t, 'kV', unom_t)

    # vetv topology parameter
    ktr = np.array(line_flows.ktr).reshape(line_n, 1)
    ts = np.ones(line_n).reshape(line_n, 1)
    is_trans = line_flows.type == 1
    ts[is_trans] = ktr[is_trans]
    u_t = u_t / ts

    # shunt conductance
    # for ordinary lines equivalent scheme P-like
    G = np.array(line_flows.g).reshape(line_n, 1)
    G_f = G / 2 * 1e-6
    G_t = G / 2 * 1e-6
    B = np.array(line_flows.b).reshape(line_n, 1)
    B_f = B / 2 * 1e-6
    B_t = B / 2 * 1e-6

    # for transformators equivalent scheme is G-like
    G_f[is_trans] = G[is_trans] * 1e-6
    G_t[is_trans] = 0.0
    B_f[is_trans] = B_f[is_trans] * 1e-6
    B_t[is_trans] = 0.0

    # conductance
    R = np.array(line_flows.r).reshape(line_n, 1)
    X = np.array(line_flows.x).reshape(line_n, 1)
    # idx = line_flows.index[(np.abs(line_flows.r) < 1e-6) & (np.abs(line_flows.x) < 1e-6)]
    # X[idx] = 0.1
    g = R / (R ** 2 + X ** 2)
    b = -X / (R ** 2 + X ** 2)

    # line voltage angle deltas
    dd = np.array(u_deltas.u_angle).reshape(line_n, 1)

    # reactive power flows
    Q_f = b * (u_f**2 - u_f * u_t * np.cos(dd)) - g * (u_f * u_t * np.sin(dd)) + B_f * u_f**2
    Q_t = b * (u_t**2 - u_f * u_t * np.cos(dd)) + g * (u_f * u_t * np.sin(dd)) + B_t * u_t**2

    return Q_f, Q_t


