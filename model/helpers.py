import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple


def convert_to_relative_units(data: np.ndarray, unit: str, v_base: np.ndarray):
    """
    Convert values from absolute to per-unit values
    :param data: source data
    :param unit: unit type
    :param v_base: base value for scaling
    :return:
    """

    if unit == 'MW':
        return data / 100.0
    elif unit == 'Ohm':
        return data * 100.0 / (v_base ** 2)
    elif unit == 'kV':
        return data / v_base
    elif unit == 'µS':
        return data * (v_base ** 2) / 100 * 1e-6
    else:
        raise ValueError(f'Incorrect unit for conversion: {unit}')


def convert_to_absolute_units(data: np.ndarray, unit: str, v_base: np.ndarray):
    """
    Convert values from per-unit values to absolute
    :param data: source data
    :param unit: unit type
    :param v_base:
    :return:
    """
    """Convert values from per-unit values to absolute"""
    if unit == 'MW':
        return data * 100.0
    elif unit == 'Ohm':
        return data / 100.0 * (v_base ** 2)
    elif unit == 'kV':
        return data * v_base
    elif unit == 'µS':
        return data / (v_base ** 2) * 100 / 1e-6
    else:
        raise ValueError(f'Incorrect unit for conversion: {unit}')


def align_voltage_level(unom: np.ndarray):
    """
    Aligns voltage to with standard values
    :param unom: initial voltage value
    :return:
    """
    """Align node base voltage level to standardized values"""
    voltage = np.array([2.8, 4.3, 6.4, 9.6, 14.4, 21.6, 32.4, 48.7, 73, 110, 150, 220, 330, 500, 750, 1150])
    unom_res = unom.copy()
    for i in range(len(voltage)):
        unom_res[np.abs(unom - voltage[i]) <= 0.21 * voltage[i]] = voltage[i]
    return unom_res


def sp_assign(A, I, J, B):
    """
    Assign matrix B to block of matrix A
    :param A: source matrix
    :param I: row indices
    :param J: column indices
    :param B: matrix to insert in A
    :return:
    """
    """A[I, J] = B"""

    A = sp_set_zero(A, I, J)
    I = np.asarray(I)
    J = np.asarray(J)
    B = B.tocsc()

    #   sort indices
    gI = np.argsort(I)
    gJ = np.argsort(J)
    I = I[gI]
    J = J[gJ]
    B = B[gI, :][:, gJ]

    va = A.tocoo()
    vb = B.tocoo()

    #   translate B-coordinates to A-coords
    br = vb.row.copy()
    bc = vb.col.copy()

    br = I[br]
    bc = J[bc]

    res = sp.coo_matrix((np.hstack([va.data, vb.data]), (np.hstack([va.row, br]), np.hstack([va.col, bc]))),
                        shape=A.shape)
    return res.tocsc()


def sp_set_zero(A, I, J):
    """A[I, J] = 0"""
    v = A.tocoo()
    to_null = np.in1d(v.row, I) & np.in1d(v.col, J)
    v.data[to_null] = 0
    return v.tocsc()


def power_system_connectivity(node: pd.DataFrame, vetv: pd.DataFrame):
    """
    For each node finds balance node to which it is connected,
    also checking that all nodes are connected to strictly one balance node.
    :param node: nodes
    :param vetv: lines
    :return:
    """

    bus_index = node['node'].argsort()
    nf = node['node'].searchsorted(vetv['node_from'], sorter=bus_index)
    nt = node['node'].searchsorted(vetv['node_to'], sorter=bus_index)

    bus_n = node.shape[0]
    sw_bus = node.index[node.type == 0].tolist()

    adj = sp.coo_matrix((np.ones(len(nf)), (nf, nt)), shape=(bus_n, bus_n)).astype(int) \
            + sp.coo_matrix((np.ones(len(nf)), (nt, nf)), shape=(bus_n, bus_n)).astype(int)

    bus_conn = np.zeros(shape=bus_n)
    for i in range(0, len(sw_bus)):
        is_connected = np.zeros(shape=bus_n)
        new_is_connected = np.zeros(shape=bus_n)
        new_is_connected[i] = 1

        while any(new_is_connected != is_connected):
            is_connected = new_is_connected
            new_is_connected = is_connected + adj * is_connected
            new_is_connected[new_is_connected > 0] = 1

        if any(bus_conn[is_connected > 0]):
            print('More than one balance bus in connected area!')
            raise

        bus_conn[new_is_connected > 0] = i

    line_conn = np.zeros(vetv.shape[0])
    for i in range(0, len(sw_bus)):
        line_conn[np.isin(nf, np.where(bus_conn == i))] = i;
        line_conn[np.isin(nt, np.where(bus_conn == i))] = i;

    return bus_conn, line_conn


def calc_z_agg(z: pd.Series) -> complex:

    cond = 1.0 / z

    return 1.0 / cond.sum()


def vetv_equiv(vetv: pd.DataFrame) -> pd.DataFrame:
    """
    Create power system without parallel branches. All line params substitute with equivalent values.
    :param node: node topology data
    :param vetv: line topology data
    :return:
    """

    vetv['z'] = vetv['r'] + 1j * vetv['x']
    is_zero = vetv['z'].abs() == 0
    vetv.loc[is_zero, 'z'] = 1e-5j
    vetv_new = vetv.groupby(by=['node_from', 'node_to']).agg({'pnum': 'count',
                                                              'type': 'min',
                                                              'z': calc_z_agg,
                                                              'g': 'sum',
                                                              'b': 'sum',
                                                              'b_from': 'sum',
                                                              'b_to': 'sum',
                                                              'ktr': 'max',
                                                              'kti': 'max',
                                                              'p_from': 'sum',
                                                              'p_to': 'sum'})
    vetv_new['r'] = np.real(vetv_new['z'])
    vetv_new['x'] = np.imag(vetv_new['z'])

    return vetv_new.reset_index()














