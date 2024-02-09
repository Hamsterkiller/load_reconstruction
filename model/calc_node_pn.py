import numpy
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix

from .power_system import PowerSystem, convert_to_relative_units


def calc_node_pn(ps: PowerSystem, node_pg: pd.DataFrame) -> np.ndarray:
    """
    Calculate node loads.
    :param ps - instance of the PowerSystem class
    :param parallel_flows - line flow values by parallel
    :param node_pg - pg values for nodes
    :return:
    """

    N_f = coo_matrix((numpy.ones(ps.line_n), (ps.node_from, range(0, ps.line_n))), shape=(ps.bus_n, ps.line_n)).astype(bool)
    N_t = coo_matrix((numpy.ones(ps.line_n), (ps.node_to, range(0, ps.line_n))), shape=(ps.bus_n, ps.line_n)).astype(bool)

    flow_from = np.array(ps.vetv.flow).reshape(ps.line_n, 1)
    flow_to = np.array(ps.vetv.flow_to).reshape(ps.line_n, 1)

    pg = convert_to_relative_units(np.array(node_pg.pg).reshape(node_pg.shape[0], 1), 'MW', np.ones(node_pg.shape[0]))

    pn = -1 * (N_f * flow_from + N_t * flow_to - pg)

    return pn



