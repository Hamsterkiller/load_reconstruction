import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .power_system import PowerSystem


def calc_rgm_newton(ps: PowerSystem, p: np.ndarray, q: np.ndarray, verbose=False) -> list[np.ndarray, float, bool]:
    """Calculate power flows using provided nodal active/reactive power injections"""

    def error_fun(a, V):
        Vbus = V * np.exp(1j * a)
        Sbus = Vbus * np.conj(ps.Ybus * Vbus)
        P = Sbus.real
        Q = Sbus.imag

        P -= p
        Q -= q

        Q[ps.pv_bus] = 0

        err_p = np.hstack([P[ps.pv_bus], P[ps.pq_bus]])
        err_q = Q[ps.pq_bus]

        max_v = V.max()
        min_v = V.min()

        delta = (ps.Cf - ps.Ct) * a
        delta[delta > np.pi] -= 2 * np.pi
        delta[delta < -np.pi] += 2 * np.pi
        max_delta_idx = np.abs(delta).argmax()

        sw_pg = P[ps.sw_bus].sum()

        return err_p, err_q, min_v, max_v, delta[max_delta_idx], sw_pg

    i = 0
    converged = False
    normF = 1e+8
    tol = 1e-6
    max_iter = 10
    Vbus = ps.get_init_voltages()

    a = np.angle(Vbus)
    V = np.abs(Vbus)

    err_p, err_q, min_v, max_v, max_delta, sw_pg = error_fun(a, V)
    if verbose:
        sys.stdout.write('\n it         err_p          err_q          min_v          max_v          max_delta      sw_pg')
        sys.stdout.write('\n----  --------------------------------------------------------------------------------------')
        sys.stdout.write('\n%3d        %10.3e     %10.3e     %10f     %10f     %10f     %10.3e' % (
            i, np.linalg.norm(err_p, np.Inf), np.linalg.norm(err_q, np.Inf), min_v, max_v, max_delta, sw_pg))

    pvpq = np.hstack([ps.pv_bus, ps.pq_bus])
    pv = ps.pv_bus
    pq = ps.pq_bus

    #   Do Newton iterations
    while not converged and i < max_iter:
        i = i + 1
        Vbus = V * np.exp(1j * a)
        dS_dVa, dS_dVm = ps.dSbus_dV(Vbus)

        j11 = dS_dVa[pvpq, :][:, pvpq].real + 1e-6 * sp.eye(len(pvpq))
        j12 = dS_dVm[pvpq, :][:, pq].real

        j21 = dS_dVa[pq, :][:, pvpq].imag
        j22 = dS_dVm[pq, :][:, pq].imag + 1e-6 * sp.eye(len(pq))

        J = sp.bmat([[j11, j12],
                     [j21, j22]]).tocsc()

        dX = -1 * spla.spsolve(J, np.hstack([err_p, err_q]))

        a[pv] = a[pv] + dX[0:len(pv)]
        a[pq] = a[pq] + dX[len(pv):len(pv) + len(pq)]
        V[pq] = V[pq] + dX[len(pv) + len(pq):]

        Vbus = V * np.exp(1j * a)
        V = np.abs(Vbus)
        a = np.angle(Vbus)

        err_p, err_q, min_v, max_v, max_delta, sw_pg = error_fun(a, V)
        normF = np.linalg.norm(np.hstack([err_p, err_q]), np.Inf)

        if verbose:
            sys.stdout.write('\n%3d        %10.3e     %10.3e     %10f     %10f     %10f     %10.3e' % (
                i, np.linalg.norm(err_p, np.Inf), np.linalg.norm(err_q, np.Inf), min_v, max_v, max_delta, sw_pg))

        if normF < tol:
            converged = True

    if verbose:
        if converged:
            sys.stdout.write('\nConverged!\n')
        else:
            sys.stdout.write('\nFailed!\n')

    return Vbus, normF, converged
