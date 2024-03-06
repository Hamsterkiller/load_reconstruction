import numpy
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix

from .power_system import PowerSystem, convert_to_relative_units


def calc_node_pn(node: pd.DataFrame, vetv: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate node active power injections.
    :param node - table with node topology
    :param vetv - table with line topology and power flows
    :return:
    """

    bus_n = node.shape[0]
    line_n = vetv.shape[0]
    node_df = node.sort_values(by=['node']).copy()
    line_df = vetv.sort_values(by=['node_from', 'node_to']).copy()
    bus_index = node_df['node'].argsort()
    nf = node_df['node'].searchsorted(line_df['node_from'], sorter=bus_index)
    nt = node_df['node'].searchsorted(line_df['node_to'], sorter=bus_index)
    pf = line_df.p_from.values
    pt = line_df.p_to.values

    N_f = coo_matrix((numpy.ones(line_n), (nf, range(0, line_n))), shape=(bus_n, line_n)).astype(bool)
    N_t = coo_matrix((numpy.ones(line_n), (nt, range(0, line_n))), shape=(bus_n, line_n)).astype(bool)

    flow_from = np.array(pf).reshape(line_n, 1)
    flow_to = np.array(pt).reshape(line_n, 1)

    pg = N_f * flow_from + N_t * flow_to

    pg_df = pd.DataFrame(node[['node']])
    pg_df['pg'] = pg

    return pg_df



