import numpy as np
import networkx as nx


def create_system_graph(topo_node: list, topo_vetv: list, node_u: list) -> nx.Graph:
    """
    Construct instance of system graph with topology information for each line and node including node voltage values.
    :param topo_node: node topology data
    :param topo_vetv: line topology data
    :param node_u:
    :return:
    """

    g = nx.Graph()
    for row in topo_vetv:
        g.add_edge(row.get('node_from'), row.get('node_to'))

    # add U and Unom attributes for each node
    node_u_dict = {row['node']: row['u'] for row in node_u}
    node_unom_dict = {row['node']: row['unom'] for row in topo_node}
    nx.classes.function.set_node_attributes(g, node_unom_dict, name='unom')
    nx.classes.function.set_node_attributes(g, node_u_dict, name='u')

    # add ktr attribute for each line of the graph
    vetv_ktr_dict = {(row['node_from'], row['node_to']): {'ktr': row['ktr'],
                                                          'node_from': row['node_from'],
                                                          'node_to': row['node_to'],
                                                          'p_from': row['p_from'],
                                                          'p_to': row['p_to'],
                                                          'type': row['type'],
                                                          'gsh': row['g'],
                                                          'bsh': row['b'],
                                                          'r': row['r'],
                                                          'x': row['x']
                                                        }
                                                            for row in topo_vetv}
    nx.classes.function.set_edge_attributes(g, vetv_ktr_dict)

    return g


def find_nearest_st_voltage(u):
    """
    Finds nearest level of standard voltage values
    :param u: source voltage value
    :return:
    """
    voltage_levels = [2.8, 4.3, 6.4, 9.6, 14.4, 21.6, 32.4, 48.7, 73, 110, 150, 220, 330, 500, 750, 1150]
    deltas = [np.abs(u - voltage_levels[i]) for i in range(0, len(voltage_levels))]
    index_min = min(range(len(deltas)), key=deltas.__getitem__)

    return deltas[index_min], voltage_levels[index_min]


def solve_flow_equations(uf: float,
                         ut: float,
                         type: int,
                         pf: float,
                         pt: float,
                         ktr: float,
                         r: float,
                         x: float,
                         gsh: float,
                         bsh: float):
    """
    Solve system of two quadratic equations for active power flows:
    P_f = g*Uf**2 - g*Uf*Ut*cos(dd) + b*Uf*Ut*sin(dd) + Uf**2*G_f
    P_t = g*Ut**2 - g*Uf*Ut*cos(dd) - b*Uf*Ut*sin(dd) + Ut**2*G_t
    1. First, get sin(dd) and cos(dd) as F(Ut).
    2. Then using the equation of sin(dd)**2 + cos(dd)**2 = 1 get Ut.
    3. Place Ut back in formulas for sin and cos.
    This system of equalities was solved analytically, so some auxiliary variables are introduced (like a1, b1. etc.)
    :param uf: voltage module at node_from
    :param ut: voltage module at node_to
    :param ut: type of the line
    :param pf: active power flow at node_from
    :param pt: active power flow at node_to
    :param ktr: transformation coefficient
    :param r: impedance
    :param x: reactance
    :param g: active conductance
    :param b: reactive conductance
    :return: voltage angle delta for line and unknown voltage module for one of the nodes
    """

    # if voltage modules for both nodes are unknown - return default flag-values
    if (uf < 0) & (ut < 0):
        return -1, -1

    # calculate line ts and conductance values
    if type != 1:
        ktr = 1
    if np.abs(ktr) < 1e-5:
        ktr = 1

    ut = ut / ktr

    # for switches return u_unknown=u_known, dd=1e-5
    if type == 2:
        u_switch = uf * (1 - 1e-3)
        if uf < 0:
            u_switch = ut * (1 + 1e-3)
        return {'u': u_switch, 'dd': 1e-5}

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
    if np.abs(R + 1j * X) < 1e-6:
        X = 0.04 / 100

    g = R / (R ** 2 + X ** 2)
    b = -X / (R ** 2 + X ** 2)

    # for lines, which both g and G are near to zero - return default values
    if (np.abs(G) < 1e-6) & (np.abs(g) < 1e-6):
        u_strange = uf * (1 - 1e-3)
        if uf < 0:
            u_strange = ut * (1 + 1e-3)
        return {'u': u_strange, 'dd': 1e-5}

    # there's two possible cases (uf is unknown and ut is unknown)]
    solution = {}
    if (ut < 0) & (uf > 0):

        # calc auxiliary variables
        a1 = G_t + g
        b1 = g * uf**2 + G_f * uf**2 - pf - pt
        c1 = 2 * uf * g
        a2 = a1
        b2 = pf - pt - g * uf**2 - G_f * uf**2
        c2 = 2 * b * uf
        k = a1**2 * c2**2 + a2**2 * c1**2
        m = 2 * a1 * b1 * c2**2 + 2 * a2 * b2 * c1**2 - c1**2 * c2**2
        n = b2**2 * c1**2 + b1**2 * c2**2

        # calc roots of square equation finding ut value
        D = np.abs(m**2 - 4 * k * n)
        # if D < 0:
        #     print(f"NEGATIVE DESCRIMINANT VALUE!")
        #     return {'u': uf * ktr * (1 - 1e-3), 'dd': 1e-3}
        x1 = (-m + np.sqrt(D)) / (2 * k)
        x2 = (-m - np.sqrt(D)) / (2 * k)

        # roots sanity check (ut * ktr must be value from 1e-3 to 1e3)
        # take solution, which is nearer to standard u values
        if (x1 > 1e-3) | (x2 > 1e-3):
            x = []
            if x1 > 1e-3:
                x.append(np.sqrt(x1))
            if x2 > 1e-3:
                x.append(np.sqrt(x2))
            deltas = []
            for xi in x:
                delta, _ = find_nearest_st_voltage(xi * ktr)
                deltas.append(delta)
            index_min = min(range(len(deltas)), key=deltas.__getitem__)
            ut = x[index_min] * ktr
            solution['u'] = ut

            # calc angle value
            dd = np.arccos(ktr * (a1 * ut**2 + b1) / (c1 * ut))
            solution['dd'] = dd
        else:
            raise Exception(f"Strange values for ut: {x1} and {x2}!")

    elif (uf < 0) & (ut > 0):
        # calc auxiliary variables
        a1 = G_f + g
        b1 = g * ut ** 2 + G_t * ut ** 2 - pf - pt
        c1 = 2 * ut * g
        a2 = a1
        b2 = ut**2 * G_t + ut**2 * g + pf - pt
        c2 = 2 * b * ut
        k = a1 ** 2 * c2 ** 2 + a2 ** 2 * c1 ** 2
        m = 2 * a1 * b1 * c2 ** 2 - 2 * a2 * b2 * c1 ** 2 - c1 ** 2 * c2 ** 2
        n = b2 ** 2 * c1 ** 2 + b1 ** 2 * c2 ** 2

        # calc roots of square equation finding ut value
        D = np.abs(m ** 2 - 4 * k * n)
        # if D < 0:
        #     print(f"NEGATIVE DESCRIMINANT VALUE!")
        #     return {'u': ut * ktr * (1 + 1e-3), 'dd': 1e-3}
        x1 = (-m + np.sqrt(D)) / (2 * k)
        x2 = (-m - np.sqrt(D)) / (2 * k)

        # roots sanity check (ut * ktr must be value from 1e-3 to 1e3)
        # take solution, which is nearer to standard u values
        if (x1 > 1e-3) | (x2 > 1e-3):
            x = []
            if x1 > 1e-3:
                x.append(np.sqrt(x1))
            if x2 > 1e-3:
                x.append(np.sqrt(x2))
            deltas = []
            for xi in x:
                delta, _ = find_nearest_st_voltage(xi)
                deltas.append(delta)
            index_min = min(range(len(deltas)), key=deltas.__getitem__)
            uf = x[index_min]
            solution['u'] = uf

            # calc angle value
            dd = np.arccos(ktr * (a1 * uf ** 2 + b1) / (c1 * uf))
            solution['dd'] = dd
    else:
        raise Exception("Invalid voltage values passed!")

    return solution


def calc_unknown_u_modules(g: nx.Graph):
    """
    Calculate unknown node voltage values.
    Let N - number of lines with at least one unknown voltage (for '_from' or '_to' node).
    Then:
    While N > 0:
        1) Calc N;
        2) For each line in L_N:
    While
    :param g: graph object of the system
    :return: void
    """

    # find initial list of edges with at least one unknown u value
    unknown_u_edges = []
    for nf, nt, v in g.edges(data=True):
        if (g.nodes[nf].get('u', -1) < 0) | (g.nodes[nt].get('u', -1) < 0):
            unknown_u_edges.append((nf, nt, v))

    while unknown_u_edges:
        for nf, nt, v in unknown_u_edges:
            # print(f"{nf} - {nt}")
            if (g.nodes[v['node_from']].get('u', -1) > 0) & (g.nodes[v['node_to']].get('u', -1) > 0):
                unknown_u_edges.remove((nf, nt, v))
            if (g.nodes[v['node_from']].get('u', -1) < 0) & (g.nodes[v['node_to']].get('u', -1) > 0):
                solution = solve_flow_equations(g.nodes[v['node_from']].get('u', -1),
                                     g.nodes[v['node_to']].get('u', -1),
                                     g.edges[nf, nt].get('type'),
                                     g.edges[nf, nt].get('p_from'),
                                     g.edges[nf, nt].get('p_to'),
                                     g.edges[nf, nt].get('ktr'),
                                     g.edges[nf, nt].get('r'),
                                     g.edges[nf, nt].get('x'),
                                     g.edges[nf, nt].get('gsh'),
                                     g.edges[nf, nt].get('bsh'))
                g.nodes[v['node_from']]['u'] = solution['u']
                g.edges[nf, nt]['dd'] = solution['dd']
                print(f"Unknown voltage module = {solution['u']}, angle = {solution['dd']}, type = {v['type']}, ktr = {v['ktr']}, ut = {g.nodes[v['node_to']]['u']}")
                unknown_u_edges.remove((nf, nt, v))
            if (g.nodes[v['node_from']].get('u', -1) > 0) & (g.nodes[v['node_to']].get('u', -1) < 0):
                solution = solve_flow_equations(g.nodes[v['node_from']].get('u', -1),
                                     g.nodes[v['node_to']].get('u', -1),
                                     g.edges[nf, nt].get('type'),
                                     g.edges[nf, nt].get('p_from'),
                                     g.edges[nf, nt].get('p_to'),
                                     g.edges[nf, nt].get('ktr'),
                                     g.edges[nf, nt].get('r'),
                                     g.edges[nf, nt].get('x'),
                                     g.edges[nf, nt].get('gsh'),
                                     g.edges[nf, nt].get('bsh'))
                g.nodes[v['node_to']]['u'] = solution['u']
                g.edges[nf, nt]['dd'] = solution['dd']
                print(f"Unknown voltage module = {solution['u']}, angle = {solution['dd']}, type = {v['type']}, ktr = {v['ktr']}, uf = {g.nodes[v['node_from']]['u']}")
                unknown_u_edges.remove((nf, nt, v))

    print("Finished!")
