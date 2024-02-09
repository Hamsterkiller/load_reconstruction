import numpy
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix

from .power_system import PowerSystem
from .helpers import convert_to_relative_units


def calc_q_injections(ps: PowerSystem, line_flows: pd.DataFrame, q_f: pd.DataFrame, q_t: pd.DataFrame) -> np.ndarray:

    """
    Calculate qg injections in each node.
    :param line_flows: table with line flows data
    :param q_f: flow
    :param q_t:
    :return:
    """

    flows_df = line_flows[['node_from', 'node_to', 'pnum']]
    flows_df['q_from'] = q_f
    flows_df['q_to'] = q_t

    N_f = coo_matrix((numpy.ones(ps.line_n), (ps.node_from, range(0, ps.line_n))), shape=(ps.bus_n, ps.line_n)).astype(
        bool)
    N_t = coo_matrix((numpy.ones(ps.line_n), (ps.node_to, range(0, ps.line_n))), shape=(ps.bus_n, ps.line_n)).astype(
        bool)

    flow_from = q_f.reshape(ps.line_n, 1)
    flow_to = q_t.reshape(ps.line_n, 1)

    qg = -1 * (N_f * flow_from + N_t * flow_to)

    return qg

