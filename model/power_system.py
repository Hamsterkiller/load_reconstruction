
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .helpers import convert_to_relative_units, align_voltage_level, sp_assign


class PowerSystem:
    """
    Power system model and modeling utilities
    """

    bus_type_sw = 0
    bus_type_pq = 1

    node: pd.DataFrame
    vetv: pd.DataFrame

    #   Start node indices for branches
    node_from: np.ndarray
    #   End node indices for branches
    node_to: np.ndarray

    pv_bus: np.ndarray
    pq_bus: np.ndarray
    sw_bus: np.ndarray

    bus_n: int
    line_n: int

    y: np.ndarray
    ts: np.ndarray
    y_br_f: np.ndarray
    y_br_t: np.ndarray

    y_ff: np.ndarray
    y_ft: np.ndarray
    y_tf: np.ndarray
    y_tt: np.ndarray

    Ybus: sp.csc_matrix
    Yf: sp.csc_matrix
    Yt: sp.csc_matrix
    Ysh: sp.csc_matrix
    Cf: sp.csc_matrix
    Ct: sp.csc_matrix

    Bbus: sp.csc_matrix
    Bf: sp.csc_matrix

    Pf_inj: np.ndarray
    Pbus_inj: np.ndarray

    def __init__(self, node: pd.DataFrame, vetv: pd.DataFrame):
        """
        Initialize power system model instance
        :param node: topo_node table of the following structure: node type unom vzd gsh bsh qn
        :param vetv: topo_vetv table of the following structure: node_from node_to pnum r x g b b_from b_to ktr kti
        """

        self.node = node.sort_values(by=['node']).copy()
        self.vetv = vetv.sort_values(by=['node_from', 'node_to', 'pnum']).copy()

        # self.node = node.copy()
        # self.vetv = vetv.copy()

        if 'b_from' not in self.vetv.columns:
            self.vetv['b_from'] = self.vetv['b'] / 2
            self.vetv['b_to'] = self.vetv['b'] / 2

        if 'g_from' not in self.vetv.columns:
            self.vetv['g_from'] = self.vetv['g'] / 2
            self.vetv['g_to'] = self.vetv['g'] / 2

        # self.rotate_transformers()

        bus_index = self.node['node'].argsort()
        self.node_from = self.node['node'].searchsorted(self.vetv['node_from'], sorter=bus_index)
        self.node_to = self.node['node'].searchsorted(self.vetv['node_to'], sorter=bus_index)

        bus_types = node['type'].values
        self.pv_bus = self.node['node'].searchsorted(self.node['node'][bus_types > 1], sorter=bus_index)
        self.pq_bus = self.node['node'].searchsorted(self.node['node'][bus_types == 1], sorter=bus_index)
        self.sw_bus = self.node['node'].searchsorted(self.node['node'][bus_types == 0], sorter=bus_index)

        self.bus_n = len(self.node)
        self.line_n = len(self.vetv)

        assert (len(self.pv_bus) + len(self.pq_bus) + len(self.sw_bus) == self.bus_n)

        self.convert_units()
        self.fix_zero_impedances()
        self.create_pf_matrices()

    def convert_units(self):
        """Align voltage levels and convert all units to per-unit system"""

        #   Align voltage levels
        # self.node['unom_orig'] = self.node['unom']
        # self.node['unom'] = align_voltage_level(self.node['unom'])

        #   Convert all units to per-unit values
        self.node['qn'] = convert_to_relative_units(self.node['qn'], 'MW', self.node['unom'])
        self.node['vzd'] = convert_to_relative_units(self.node['vzd'], 'kV', self.node['unom'])

        self.node['gsh'] = convert_to_relative_units(self.node['gsh'], 'µS', self.node['unom'])
        self.node['bsh'] = -convert_to_relative_units(self.node['bsh'], 'µS', self.node['unom'])

        #   Incorrect voltage magnitude for PV/SW-type buses, fix it
        self.node.loc[(self.node['type'] != self.bus_type_pq) & (self.node['vzd'] < 0.5), 'vzd'] = 1

        # unom_max = np.maximum(self.node['unom'].values[self.node_from], self.node['unom'].values[self.node_to])
        # unom_min = np.minimum(self.node['unom'].values[self.node_from], self.node['unom'].values[self.node_to])
        unom_from = self.node['unom'].values[self.node_from]
        unom_to = self.node['unom'].values[self.node_to]

        self.vetv['node_from_orig'] = self.vetv['node_from']
        self.vetv['node_to_orig'] = self.vetv['node_to']

        self.vetv['ts'] = self.vetv['ktr'].values + 1j * self.vetv['kti'].values
        is_tr = np.abs(self.vetv['ts']) > 0
        vf = self.node.unom.iloc[self.node_from].values
        vt = self.node.unom.iloc[self.node_to].values
        is_tr[vf < vt] = False
        self.vetv.loc[is_tr, 'node_from'] = self.vetv.loc[is_tr, 'node_to_orig']
        self.vetv.loc[is_tr, 'node_to'] = self.vetv.loc[is_tr, 'node_from_orig']
        self.vetv.loc[is_tr, 'p_from'] = self.vetv.loc[is_tr, 'p_to']
        self.vetv.loc[is_tr, 'p_to'] = self.vetv.loc[is_tr, 'p_from']
        self.vetv.sort_values(['node_from', 'node_to', 'pnum'], inplace=True)
        self.vetv.index = list(range(0, self.vetv.shape[0]))
        bus_index = self.node['node'].argsort()
        self.node_from = self.node['node'].searchsorted(self.vetv['node_from'], sorter=bus_index)
        self.node_to = self.node['node'].searchsorted(self.vetv['node_to'], sorter=bus_index)

        self.vetv['r'] = convert_to_relative_units(self.vetv['r'], 'Ohm', unom_from)
        self.vetv['x'] = convert_to_relative_units(self.vetv['x'], 'Ohm', unom_from)

        self.vetv['g'] = convert_to_relative_units(self.vetv['g'], 'µS', unom_from)
        self.vetv['g_from'] = convert_to_relative_units(self.vetv['g_from'], 'µS', unom_from)
        self.vetv['g_to'] = convert_to_relative_units(self.vetv['g_to'], 'µS', unom_to)

        self.vetv['b'] = -convert_to_relative_units(self.vetv['b'], 'µS', unom_from)
        self.vetv['b_from'] = -convert_to_relative_units(self.vetv['b_from'], 'µS', unom_from)
        self.vetv['b_to'] = -convert_to_relative_units(self.vetv['b_to'], 'µS', unom_to)

        self.vetv['u_from'] = convert_to_relative_units(self.vetv['u_from'], 'kV', unom_from)
        self.vetv['u_to'] = convert_to_relative_units(self.vetv['u_to'], 'kV', unom_to)
        self.vetv['p_from'] = convert_to_relative_units(self.vetv['p_from'], 'MW', unom_from)
        self.vetv['p_to'] = convert_to_relative_units(self.vetv['p_to'], 'MW', unom_to)
        # self.vetv['ktr'] = self.vetv['ktr'] / unom_to * unom_from
        # self.vetv['kti'] = self.vetv['kti'] / unom_to * unom_from

    def rotate_transformers(self):
        """Rotate transformers"""

        #   We use \Gamma-kind substitution scheme for transformers and will require branch
        #   direction from the upper voltage to the lower
        bus_index = self.node['node'].argsort()
        node_from = self.node['node'].searchsorted(self.vetv['node_from'], sorter=bus_index)
        node_to = self.node['node'].searchsorted(self.vetv['node_to'], sorter=bus_index)
        rotate_transformers = (self.vetv['ktr'].values > 0) & (
                self.node.loc[node_from, 'unom'].values < self.node.loc[node_to, 'unom'].values)
        self.vetv['is_rotated'] = rotate_transformers
        self.vetv.loc[rotate_transformers, 'ktr'] = 1 / self.vetv.loc[rotate_transformers, 'ktr']
        nf = self.vetv['node_from'].copy()
        nt = self.vetv['node_to'].copy()
        self.vetv.loc[rotate_transformers, 'node_from'] = nt[rotate_transformers]
        self.vetv.loc[rotate_transformers, 'node_to'] = nf[rotate_transformers]

    def fix_zero_impedances(self):
        """Fix close-to-zero branch impedances"""

        #   Derived experimentally from the Rastr model
        zero_idx = np.abs(self.vetv['r'] + 1j * self.vetv['x']) < 1e-6
        self.vetv.loc[zero_idx, 'x'] = 0.04 / 100

    def create_pf_matrices(self):
        # shunt_b = self.node['bsh'].values.copy()
        # shunt_g = self.node['gsh'].values.copy()
        shunt_b = np.zeros(self.bus_n)
        shunt_g = np.zeros(self.bus_n)
        # shunt_bus = find(obj.Bus.gsh ~= 0 | obj.Bus.bsh ~= 0);
        shunt_bus = (self.node.gsh != 0.0) | (self.node.bsh != 0.0)
        shunt_b[shunt_bus] = self.node.loc[shunt_bus, 'bsh'].values.copy()
        shunt_g[shunt_bus] = self.node.loc[shunt_bus, 'gsh'].values.copy()

        #   Reactive loads at PQ buses (modelled as shunt with specified Qn at nominal voltage)
        shunt_b[self.pq_bus] = shunt_b[self.pq_bus] - self.node['qn'].values[self.pq_bus]
        #   Reactive loads at PV buses (modelled as shunt with specified Qn at nominal voltage)
        shunt_b[self.pv_bus] = shunt_b[self.pv_bus] - self.node['qn'].values[self.pv_bus]

        #   Branch admittances to the ground
        y_br = self.vetv['g'].values + 1j * self.vetv['b'].values
        # y_br_f = self.vetv['g_from'].values + 1j * self.vetv['b_from'].values
        # y_br_t = self.vetv['g_to'].values + 1j * self.vetv['b_to'].values
        y_br_f = self.vetv['g'].values + 1j * self.vetv['b'].values
        y_br_t = self.vetv['g'].values + 1j * self.vetv['b'].values

        #   Transformer admittances
        ts = self.vetv.ts
        # is_tr = np.abs(ts) > 0
        is_tr = self.vetv.type == 1
        is_line = self.vetv.type == 0
        ts[~is_tr] = 1.0
        y_br_f[is_tr] = 0.0
        y_br_t[is_tr] = 2 * y_br[is_tr]
        y_br_f[is_line] = self.vetv.loc[is_line, 'g'].values + 2j * self.vetv.loc[is_line, 'b_from'].values
        y_br_t[is_line] = self.vetv.loc[is_line, 'g'].values + 2j * self.vetv.loc[is_line, 'b_to'].values
        # y_br_t = self.vetv['g'].values + 1j * self.vetv['b'].values

        #   We need it if we have different nominal voltages for different branch ends
        k = self.node['unom'].values[self.node_to] / self.node['unom'].values[self.node_from]
        ts *= k

        #   Node-branch incidence matrices
        self.Cf = sp.csc_matrix((np.ones(self.line_n), (range(self.line_n), self.node_from)), shape=(self.line_n, self.bus_n))
        self.Ct = sp.csc_matrix((np.ones(self.line_n), (range(self.line_n), self.node_to)), shape=(self.line_n, self.bus_n))

        z = (self.vetv['r'] + 1j * self.vetv['x']).values
        unomFrom = self.node.unom.iloc[self.node_from].tolist()
        unomTo = self.node.unom.iloc[self.node_to].tolist()
        baseV = np.array([np.max([unomFrom[i], unomTo[i]]) for i in range(0, self.line_n)])
        # idx_zero = abs(real(z)) + abs(imag(z)) <= 1. / baseV. ^ 2;
        idx_zero = (np.abs(z.real) + np.abs(z.imag)) <= (1.0 / baseV**2)
        z[idx_zero] = 0.04j / 100
        y = 1 / z

        #  For each branch, compute the elements of the branch admittance matrix where
        #  | If |   | Yff  Yft |   | Vf |
        #  |    | = |          | * |    |
        #  | It |   | Ytf  Ytt |   | Vt |
        # self.y_ff = y + y_br_f
        # self.y_ft = -y / ts
        # self.y_tf = -y / ts.conj()
        # self.y_tt = y / (ts * ts.conj()) + y_br_t
        ts2 = (ts * ts.conj()).real
        self.y_ff = (y + y_br_f / 2) / ts2
        self.y_ft = -y / ts.conj()
        self.y_tf = -y / ts
        self.y_tt = y + y_br_t / 2

        self.y = y
        self.ts = ts
        self.y_br_f = y_br_f
        self.y_br_t = y_br_t

        self.Yf = sp.csc_matrix((self.y_ff, (range(self.line_n), self.node_from)), shape=(self.line_n, self.bus_n)) + \
                  sp.csc_matrix((self.y_ft, (range(self.line_n), self.node_to)), shape=(self.line_n, self.bus_n))

        self.Yt = sp.csc_matrix((self.y_tf, (range(self.line_n), self.node_from)), shape=(self.line_n, self.bus_n)) + \
                  sp.csc_matrix((self.y_tt, (range(self.line_n), self.node_to)), shape=(self.line_n, self.bus_n))

        self.Ysh = sp.csc_matrix((shunt_g + 1j * shunt_b, (range(self.bus_n), range(self.bus_n))), shape=(self.bus_n, self.bus_n))

        #   https://en.wikipedia.org/wiki/Nodal_admittance_matrix
        self.Ybus = (self.Cf.T * self.Yf + self.Ct.T * self.Yt + self.Ysh).tocsc()

    def get_init_voltages(self) -> np.ndarray:
        """Calculate initial voltage guess for zero power flow"""
        V = np.ones(self.bus_n)

        #   Fixed voltage magnitudes for PV/SW-buses
        V[self.pv_bus] = self.node['vzd'].values[self.pv_bus]
        V[self.sw_bus] = self.node['vzd'].values[self.sw_bus]
        a = np.zeros(self.bus_n)

        #   Non-PQ buses
        npq = np.hstack([self.sw_bus, self.pv_bus])

        Y_pq = self.Ybus[self.pq_bus, :][:, self.pq_bus]
        Y_npq = self.Ybus[self.pq_bus, :][:, npq]

        #   Solve equations for the complex voltages: Vbus_pq = u_pq + 1j * v_pq:
        #   Y_pq * Vbus_pq + Y_npq * Vbus_npq = 0 (current balances for the each PQ-bus)
        Vbus_npq = V[npq] * np.exp(1j * a[npq])
        M = sp.bmat([[Y_pq.real, -Y_pq.imag],
                     [Y_pq.imag, Y_pq.real]]).tocsc()

        rhs = -np.hstack([(Y_npq @ Vbus_npq).real, (Y_npq @ Vbus_npq).imag])
        X = spla.spsolve(M, rhs)

        u_pq = X[:len(self.pq_bus)]
        v_pq = X[len(self.pq_bus):]

        Vbus_pq = u_pq + 1j * v_pq
        V[self.pq_bus] = np.abs(Vbus_pq)
        a[self.pq_bus] = np.angle(Vbus_pq)

        Vbus = V * np.exp(1j * a)

        return Vbus

    def flows(self, Vbus: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate power flows"""
        Ibus = self.Ybus * Vbus
        If = self.Yf * Vbus
        It = self.Yt * Vbus

        Sbus = Vbus * np.conj(Ibus)
        Sf = Vbus[self.node_from] * np.conj(If)
        St = Vbus[self.node_to] * np.conj(It)

        return Sbus, Sf, St

    def balance_jacobian(self, Vbus: np.ndarray) -> (sp.csc_matrix, sp.csc_matrix):
        """Returns two power balance jacobians w.r.t. voltage magnitudes and angles"""
        pv = self.pv_bus
        pq = self.pq_bus
        sw = self.sw_bus
        pvsw = np.hstack([pv, sw])
        pvpq = np.hstack([pv, pq])

        dS_dVa, dS_dVm = self.dSbus_dV(Vbus)

        j11 = dS_dVa[pvpq, :][:, pvpq].real + 1e-6 * sp.eye(len(pvpq))
        j12 = dS_dVm[pvpq, :][:, pq].real
        j21 = dS_dVa[pq, :][:, pvpq].imag
        j22 = dS_dVm[pq, :][:, pq].imag + 1e-6 * sp.eye(len(pq))

        J = sp.bmat([[j11, j12], [j21, j22]])  # [dP/dVa dP/dVm]
        Jsw = sp.hstack([dS_dVa[sw, :][:, pvpq].real, dS_dVm[sw, :][:, pq].real])  # [dPsw/dVa dPsw/dVm]

        return Jsw.tocsc(), J.tocsc()

    def loss_factors(self, Vbus: np.ndarray) -> sp.csc_matrix:
        """Power balance distribution factors"""
        pv = self.pv_bus
        pq = self.pq_bus
        sw = self.sw_bus
        pvsw = np.hstack([pv, sw])
        pvpq = np.hstack([pv, pq])

        Jsw, J = self.balance_jacobian(Vbus)

        #   X * [dP/dVa dP/dVm; dQ/dVa dQ/dVm] = [dPsw/dVa dPsw/dVm] <=> X * J = Jsw
        lu = spla.splu(J.tocsc())
        X = lu.solve(Jsw.T.todense(), 'T').T  # [dPsw/dP dQsw/dP]

        lpc = np.zeros((len(sw), 2 * self.bus_n))
        lpc[:, np.hstack([pvpq, self.bus_n + pq])] = X
        lpc[:, sw] = -np.eye(len(sw))
        lpc[:, self.bus_n + sw] = -np.eye(len(sw))

        return lpc

    def line_factors(self, Vbus: np.ndarray) -> sp.csc_matrix:
        """Line flows distribution factors (PTDF)"""
        pv = self.pv_bus
        pq = self.pq_bus
        sw = self.sw_bus
        pvsw = np.hstack([pv, sw])
        pvpq = np.hstack([pv, pq])

        Jsw, J = self.balance_jacobian(Vbus)

        dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm = self.dSbr_dV(Vbus)

        Jlines = sp.bmat([
            [dSf_dVa[:, pvpq].real, dSf_dVm[:, pq].real],
            [dSt_dVa[:, pvpq].real, dSt_dVm[:, pq].real]
        ]).tocsr()

        Jlines = sp.bmat([
            [dSf_dVa[:, pvpq].imag, dSf_dVm[:, pq].imag],
            [dSt_dVa[:, pvpq].imag, dSt_dVm[:, pq].imag]
        ]).tocsr()

        #   X * [dP/dVa dP/dVm; dQ/dVa dQ/dVm] = [dPsw/dVa dPsw/dVm] <=> X * J = Jsw
        # lu = spla.splu(J.tocsc())
        # X = lu.solve(Jsw.T.todense(), 'T').T # [dPsw/dP dQsw/dP]
        X = spla.spsolve(J.T.tocsc(), Jlines.T, True).T

        Jlf = sp.csc_matrix((2 * self.line_n, self.bus_n))
        Jlf = sp_assign(Jlf, np.arange(0, 2 * self.line_n), pvpq, X[:, 0:len(pvpq)])
        return Jlf

    def dSbus_dV(self, Vbus: np.ndarray):
        """Computes partial derivatives of power injection w.r.t. voltage.
        Returns two matrices containing partial derivatives of the complex bus
        power injections w.r.t voltage magnitude and voltage angle respectively
        (for all buses). If C{Ybus} is a sparse matrix, the return values will be
        also. The following explains the expressions used to form the matrices::
            S = diag(V) * conj(Ibus) = diag(conj(Ibus)) * V
        Partials of V & Ibus w.r.t. voltage magnitudes::
            dV/dVm = diag(V / abs(V))
            dI/dVm = Ybus * dV/dVm = Ybus * diag(V / abs(V))
        Partials of V & Ibus w.r.t. voltage angles::
            dV/dVa = j * diag(V)
            dI/dVa = Ybus * dV/dVa = Ybus * j * diag(V)
        Partials of S w.r.t. voltage magnitudes::
            dS/dVm = diag(V) * conj(dI/dVm) + diag(conj(Ibus)) * dV/dVm
                   = diag(V) * conj(Ybus * diag(V / abs(V)))
                                            + conj(diag(Ibus)) * diag(V / abs(V))
        Partials of S w.r.t. voltage angles::
            dS/dVa = diag(V) * conj(dI/dVa) + diag(conj(Ibus)) * dV/dVa
                   = diag(V) * conj(Ybus * j * diag(V))
                                            + conj(diag(Ibus)) * j * diag(V)
                   = -j * diag(V) * conj(Ybus * diag(V))
                                            + conj(diag(Ibus)) * j * diag(V)
                   = j * diag(V) * conj(diag(Ibus) - Ybus * diag(V))
        For more details on the derivations behind the derivative code used
        in PYPOWER information, see:
        [TN2]  R. D. Zimmerman, "AC Power Flows, Generalized OPF Costs and
        their Derivatives using Complexex Matrix Notation", MATPOWER
        Technical Note 2, February 2010.
        U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}
        @author: Ray Zimmerman (PSERC Cornell)
        """
        ib = range(self.bus_n)

        Ibus = self.Ybus @ Vbus

        diagVbus = sp.csc_matrix((Vbus, (ib, ib)))
        diagIbus = sp.csc_matrix((Ibus, (ib, ib)))
        diagVnorm = sp.csc_matrix((Vbus / np.abs(Vbus), (ib, ib)))

        dS_dVa = 1j * diagVbus @ (diagIbus - self.Ybus @ diagVbus).conj()
        dS_dVm = diagVbus @ (self.Ybus * diagVnorm).conj() + diagIbus.conj() @ diagVnorm

        return dS_dVa, dS_dVm

    def dSbr_dV(self, Vbus: np.ndarray) -> Tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]:
        """Computes partial derivatives of power flows w.r.t. voltage.
            returns four matrices containing partial derivatives of the complex
            branch power flows at "from" and "to" ends of each branch w.r.t voltage
            magnitude and voltage angle respectively (for all buses). If C{Yf} is a
            sparse matrix, the partial derivative matrices will be as well. Optionally
            returns vectors containing the power flows themselves. The following
            explains the expressions used to form the matrices::
                If = Yf * V;
                Sf = diag(Vf) * conj(If) = diag(conj(If)) * Vf
            Partials of V, Vf & If w.r.t. voltage angles::
                dV/dVa  = j * diag(V)
                dVf/dVa = sparse(range(nl), f, j*V(f)) = j * sparse(range(nl), f, V(f))
                dIf/dVa = Yf * dV/dVa = Yf * j * diag(V)
            Partials of V, Vf & If w.r.t. voltage magnitudes::
                dV/dVm  = diag(V / abs(V))
                dVf/dVm = sparse(range(nl), f, V(f) / abs(V(f))
                dIf/dVm = Yf * dV/dVm = Yf * diag(V / abs(V))
            Partials of Sf w.r.t. voltage angles::
                dSf/dVa = diag(Vf) * conj(dIf/dVa)
                                + diag(conj(If)) * dVf/dVa
                        = diag(Vf) * conj(Yf * j * diag(V))
                                + conj(diag(If)) * j * sparse(range(nl), f, V(f))
                        = -j * diag(Vf) * conj(Yf * diag(V))
                                + j * conj(diag(If)) * sparse(range(nl), f, V(f))
                        = j * (conj(diag(If)) * sparse(range(nl), f, V(f))
                                - diag(Vf) * conj(Yf * diag(V)))
            Partials of Sf w.r.t. voltage magnitudes::
                dSf/dVm = diag(Vf) * conj(dIf/dVm)
                                + diag(conj(If)) * dVf/dVm
                        = diag(Vf) * conj(Yf * diag(V / abs(V)))
                                + conj(diag(If)) * sparse(range(nl), f, V(f)/abs(V(f)))
            Derivations for "to" bus are similar.
            For more details on the derivations behind the derivative code used
            in PYPOWER information, see:
            [TN2]  R. D. Zimmerman, "AC Power Flows, Generalized OPF Costs and
            their Derivatives using Complex Matrix Notation", MATPOWER
            Technical Note 2, February 2010.
            U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}
            @author: Ray Zimmerman (PSERC Cornell)
            """
        ib = np.arange(self.bus_n)
        il = np.arange(self.line_n)

        nb = len(ib)
        nl = len(il)

        nf = self.node_from
        nt = self.node_to

        Vnorm = Vbus / abs(Vbus)

        If = self.Yf @ Vbus
        It = self.Yt @ Vbus

        diagVf = sp.csc_matrix((Vbus[nf], (il, il)))
        diagVt = sp.csc_matrix((Vbus[nt], (il, il)))
        diagIf = sp.csc_matrix((If, (il, il)))
        diagIt = sp.csc_matrix((It, (il, il)))
        diagVbus = sp.csc_matrix((Vbus, (ib, ib)))
        diagVnorm = sp.csc_matrix((Vnorm, (ib, ib)))

        # Partial derivative of S w.r.t voltage phase angle.
        dSf_dVa = 1j * (diagIf.conj() @ sp.csc_matrix((Vbus[nf], (il, nf)), (nl, nb)) - diagVf @ np.conj
            (self.Yf @ diagVbus))
        dSt_dVa = 1j * (diagIt.conj() @ sp.csc_matrix((Vbus[nt], (il, nt)), (nl, nb)) - diagVt @ np.conj
            (self.Yt @ diagVbus))

        # Partial derivative of S w.r.t. voltage amplitude.
        dSf_dVm = diagVf @ np.conj(self.Yf @ diagVnorm) + diagIf.conj() @ sp.csc_matrix((Vnorm[nf], (il, nf)), (nl, nb))
        dSt_dVm = diagVt @ np.conj(self.Yt @ diagVnorm) + diagIt.conj() @ sp.csc_matrix((Vnorm[nt], (il, nt)), (nl, nb))

        return dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm

    def d2Sbus_dV2(self, Vbus: np.ndarray, lam: np.ndarray):
        """Computes 2nd derivatives of power injection w.r.t. voltage.
        Returns 4 matrices containing the partial derivatives w.r.t. voltage angle
        and magnitude of the product of a vector C{lam} with the 1st partial
        derivatives of the complex bus power injections. Takes sparse bus
        admittance matrix C{Ybus}, voltage vector C{V} and C{nb x 1} vector of
        multipliers C{lam}. Output matrices are sparse.
        For more details on the derivations behind the derivative code used
        in PYPOWER information, see:
        [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
        their Derivatives using Complex Matrix Notation"}, MATPOWER
        Technical Note 2, February 2010.
        U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}
        @author: Ray Zimmerman (PSERC Cornell)
        """
        nb = len(Vbus)
        ib = np.arange(nb)
        Ibus = self.Ybus @ Vbus
        diaglam = sp.csc_matrix((lam, (ib, ib)))
        diagV = sp.csc_matrix((Vbus, (ib, ib)))

        A = sp.csc_matrix((lam * Vbus, (ib, ib)))
        B = self.Ybus @ diagV
        C = A @ B.conj()
        D = self.Ybus.H * diagV
        E = diagV.conj() @ (D @ diaglam - sp.csc_matrix((D @ lam, (ib, ib))))
        F = C - A * sp.csc_matrix((Ibus.conj(), (ib, ib)))
        G = sp.csc_matrix((np.ones(nb) / np.abs(Vbus), (ib, ib)))

        Gaa = E + F
        Gva = 1j * G @ (E - F)
        Gav = Gva.T
        Gvv = G @ (C + C.T) @ G

        return Gaa, Gav, Gva, Gvv

    def d2Sbr_dV2(self, Cbr: sp.csc_matrix, Ybr: sp.csc_matrix, Vbus: np.ndarray, lam: np.ndarray):
        """Computes 2nd derivatives of complex power flow w.r.t. voltage.
        Returns 4 matrices containing the partial derivatives w.r.t. voltage angle
        and magnitude of the product of a vector C{lam} with the 1st partial
        derivatives of the complex branch power flows. Takes sparse connection
        matrix C{Cbr}, sparse branch admittance matrix C{Ybr}, voltage vector C{V}
        and C{nl x 1} vector of multipliers C{lam}. Output matrices are sparse.
        For more details on the derivations behind the derivative code used
        in PYPOWER information, see:
        [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
        their Derivatives using Complex Matrix Notation"}, MATPOWER
        Technical Note 2, February 2010.
        U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}
        @author: Ray Zimmerman (PSERC Cornell)
        """
        nb = len(Vbus)
        nl = len(lam)
        ib = np.arange(nb)
        il = np.arange(nl)

        diaglam = sp.csc_matrix((lam, (il, il)))
        diagV = sp.csc_matrix((Vbus, (ib, ib)))

        A = Ybr.H @ diaglam @ Cbr
        B = diagV.conj() @ A @ diagV
        D = sp.csc_matrix(((A @ Vbus) * Vbus.conj(), (ib, ib)))
        E = sp.csc_matrix(((A.T @ Vbus.conj() * Vbus), (ib, ib)))
        F = B + B.T
        G = sp.csc_matrix((np.ones(nb) / abs(Vbus), (ib, ib)))

        Haa = F - D - E
        Hva = 1j * G * (B - B.T - D + E)
        Hav = Hva.T
        Hvv = G * F * G

        return Haa, Hav, Hva, Hvv
