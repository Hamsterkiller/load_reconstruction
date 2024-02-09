import pandas as pd
import numpy as np
from .power_system import PowerSystem, convert_to_relative_units
from .calc_node_pn import calc_node_pn
from .calc_u_deltas import calc_u_deltas
from .helpers import vetv_equiv
from .calc_unknown_u_modules import calc_unknown_u_modules, create_system_graph
from .calc_q_flows import calc_q_flows
from .calc_q_injections import calc_q_injections
from .calc_rgm_newton import calc_rgm_newton


def reconstruct_hour(hour: int, topology_data: dict[str, pd.DataFrame], src_data: dict[str, pd.DataFrame]):
    """
    Reconstruct system parameters for one hour
    :param hour:
    :param topology_data:
    :param src_data:
    :return:
    """

    node = topology_data['node']
    vetv = topology_data['vetv']

    # make actual parallels pool
    parallel_flows = src_data['parallel_flows'].query(f"hour == {hour}")
    parallel_flows.rename({'flow': 'p_from', 'flow_to': 'p_to'}, inplace=True, axis=1)

    # correct missing flow values if exist
    fidx_na = parallel_flows.index[parallel_flows.p_from.isna() | parallel_flows.p_to.isna()]
    for idx in fidx_na:
        if np.isnan(parallel_flows.iloc[idx].p_from) & (not np.isnan(parallel_flows.iloc[idx].p_to)):
            print(f"Filling missing p_from value for the line {parallel_flows.iloc[idx].node_from} "
                  f"- {parallel_flows.iloc[idx].node_to}")
            if parallel_flows.iloc[idx].p_to < 0:
                parallel_flows.at[idx, 'p_from'] = -0.999 * parallel_flows.iloc[idx].p_to
            else:
                parallel_flows.at[idx, 'p_from'] = -1.001 * parallel_flows.iloc[idx].p_to
        if not np.isnan(parallel_flows.iloc[idx].p_from) & np.isnan(parallel_flows.iloc[idx].p_to):
            print(f"Filling missing p_tovalue for the line {parallel_flows.iloc[idx].node_from} "
                  f"- {parallel_flows.iloc[idx].node_to}")
            if parallel_flows.iloc[idx].p_from > 0:
                parallel_flows.at[idx, 'p_to'] = -0.999 * parallel_flows.iloc[idx].p_from
            else:
                parallel_flows.at[idx, 'p_to'] = -1.001 * parallel_flows.iloc[idx].p_from
        else:
            raise Exception(f"Vetv {parallel_flows.iloc[idx].node_from} - {parallel_flows.iloc[idx].node_to} has empty flow fields!")

    vetv_direct = vetv.merge(right=parallel_flows, on=['node_from', 'node_to', 'pnum'], how='inner')
    vetv_inverse = vetv.merge(right=parallel_flows, left_on=['node_from', 'node_to', 'pnum'],
                              right_on=['node_to', 'node_from', 'pnum'], how='inner')
    vetv_inverse.drop(['node_from_y', 'node_to_y'], axis=1, inplace=True)
    vetv_inverse.rename({'node_from_x': 'node_from', 'node_to_x': 'node_to'}, inplace=True, axis=1)

    # remove parallels, that exist in ATS parallels flow report in both directions
    vetv_control = vetv_inverse.merge(right=vetv_direct,
                                      left_on=['node_from', 'node_to', 'pnum'],
                                      right_on=['node_to', 'node_from', 'pnum'],
                                      how='left')
    vetv_control = vetv_control.loc[vetv_control.node_from_y.isna(), ['node_from_x', 'node_to_x', 'pnum']]
    vetv_control.rename({'node_from_x': 'node_from', 'node_to_x': 'node_to'}, axis=1, inplace=True)
    vetv_inverse = vetv_inverse.merge(right=vetv_control, on=['node_from', 'node_to', 'pnum'], how='inner')

    vetv = pd.concat([vetv_direct, vetv_inverse], axis=0)
    vetv = vetv[['node_from', 'node_to', 'pnum', 'type', 'r', 'x', 'g', 'b', 'b_from',
                 'b_to', 'ktr', 'kti', 'p_from', 'p_to']]

    node_u_values = src_data['node_prices'][['node', 'u', 'hour']].query(f'hour == {hour}').drop(['hour'], axis=1)

    # remove all nodes, that are not in the topology for this hour
    all_nodes = set(vetv.node_from.unique().tolist() + vetv.node_to.unique().tolist())
    node = node[node.node.isin(all_nodes)]
    node.index = range(0, node.shape[0])

    # construct graph of the system
    vetv_eq = vetv_equiv(vetv)
    vetv_eq.loc[np.abs(vetv_eq['ktr']) < 1e-5, 'ktr'] = 1.0
    vetv_eq = vetv_eq.merge(node_u_values, left_on=['node_from'], right_on=['node'], how='left')
    vetv_eq.rename({'u': 'u_from'}, inplace=True, axis=1)
    vetv_eq.drop({'node'}, axis=1, inplace=True)
    vetv_eq = vetv_eq.merge(node_u_values, left_on=['node_to'], right_on=['node'], how='left')
    vetv_eq.rename({'u': 'u_to'}, inplace=True, axis=1)
    vetv_eq.drop({'node'}, axis=1, inplace=True)
    vetv_dict = vetv_eq.to_dict(orient='records')
    node_dict = node.to_dict(orient='records')
    node_u_dict = node_u_values.to_dict(orient='records')
    system_graph = create_system_graph(node_dict, vetv_dict, node_u_dict)

    # calculate u value for each node with unknown u
    calc_unknown_u_modules(system_graph)

    # create dataframe with u values
    u_dict_df = []
    for n, v in system_graph.nodes(data=True):
        u_dict_df.append({'node': n, 'u': v['u']})
    u_df = pd.DataFrame.from_records(u_dict_df)

    # calculate dd for each line in graph without known dd
    calc_u_deltas(system_graph)

    # node, vetv = power_system_connectivity(node, vetv)

    # initialize PowerSystem instance
    ps = PowerSystem(node, vetv)

    # correct u values of needed
    u_df = ps.node[['node', 'unom']].merge(right=u_df, on=['node'], how='left')
    strange_idx = u_df.index[(u_df.u / u_df.unom > 1.5) | (u_df.u / u_df.unom < 0.5)]
    for idx in strange_idx:
        u_df.at[idx, 'u'] = u_df.iloc[idx].unom

    # calculate gen volumes for each node
    rge_node = src_data['rge_pmin_pmax'][['rge', 'p', 'hour']]\
                                                    .merge(right=topology_data['rge'][['rge', 'node']],
                                                        on=['rge'],
                                                        how='left')\
                                                    .query(f'hour=={hour}')\
                                                    .drop(['hour'], axis=1)
    node_pg = rge_node.groupby(['node'], as_index=False).sum('p').drop(['rge'], axis=1)
    node_pg = ps.node.node.to_frame().merge(right=node_pg, on=['node'], how='left').fillna(0)
    node_pg.rename({'p': 'pg'}, axis=1, inplace=True)
    node_pg = ps.node['node'].to_frame().merge(right=node_pg, on=['node'], how='left')

    # calculate pn for each node in the system
    pn = calc_node_pn(ps, node_pg)
    idx = np.where(pn < 0)
    pn[idx] = 0.0;
    node_pn = pd.concat([node_pg.node, pd.Series(pn.reshape(-1))], axis=1)
    node_pn.columns = ['node', 'pn']

    # calculate U angle differences for each line, where both nodes have known U value
    # bus_voltages = src_data['node_prices'].loc[(~src_data['node_prices'].u.isna())
    #                                            & (src_data['node_prices'].hour == hour), ['node', 'u']]
    # line_flows = ps.vetv[['node_from', 'node_to', 'pnum', 'type', 'r', 'x', 'g', 'b', 'ktr', 'flow', 'flow_to']]
    # deltas = calc_u_deltas(u_df, line_flows)

    # calculate reactive power flows for each line in the system and qn in each node
    # Q_f, Q_t = calc_q_flows(ps, u_df, line_flows, deltas)

    # calculate qg bus injections
    # qg = calc_q_injections(ps, line_flows, Q_f, Q_t)

    # calculate regime using Newton method
    # pg = convert_to_relative_units(np.array(node_pg.pg).reshape(node_pg.shape[0], 1), 'MW', np.ones(node_pg.shape[0]))
    # pg = np.array(node_pg.pg).reshape(node_pg.shape[0], 1)
    # p = (pg - pn).reshape(-1)
    # q = qg.reshape(-1)
    # Vbus, normF, converged = calc_rgm_newton(ps, p, q, verbose=True)

    # calculate qn for each node in the system, that has published U value
    # node_qn = ps.node[['node', 'qn', 'unom']].merge(right=node_u_values, on=['node'], how='inner')
    # node_qn['qn_new'] = node_qn['qn'] * node_qn['u']**2 / node_qn['unom']**2

    # calculate qn for nodes where U values are unknown
    # nodes_with_known_u = node_qn.node.unique()
    # nodes_with_unknown_u = [n for n in ps.node['node'].tolist() if n not in nodes_with_known_u]



