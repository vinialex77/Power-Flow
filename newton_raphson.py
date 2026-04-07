import numpy as np
from mismatch import calc_power
from jacobian import build_jacobian

def newton_raphson(buses, Ybus, tol=1e-6, max_iter=20):
    n = len(buses)

    V = np.ones(n)
    theta = np.zeros(n)
    P_spec = np.zeros(n)
    Q_spec = np.zeros(n)
    slack_index = None
    pv_index = []
    pq_index = []

    for i, b in enumerate(buses):
        if b["type"] == "Slack":
            slack_index = i
            V[i] = b["V"]
            theta[i] = np.radians(b["theta"])
        elif b["type"] == "PV":
            pv_index.append(i)
            V[i] = b["V"]
            P_spec[i] = b["P"]
        elif b["type"] == "PQ":
            pq_index.append(i)
            P_spec[i] = b["P"]
            Q_spec[i] = b["Q"]

    pvpq = pv_index + pq_index
    historico_passos = []

    for iteration in range(max_iter):
        V_atual = V.copy()
        theta_atual = theta.copy()

        P_calc, Q_calc = calc_power(V, theta, Ybus)
        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc
        
        mismatch = np.concatenate([dP[pvpq], dQ[pq_index]])
        
        erro = np.max(np.abs(mismatch))
        convergiu = erro < tol

        passo_info = {
            'nu': iteration,
            'V_nu': V_atual,
            'theta_nu': theta_atual,
            'P_calc': P_calc.copy(),
            'Q_calc': Q_calc.copy(),
            'dP': dP.copy(),
            'dQ': dQ.copy(),
            'mismatch': mismatch.copy(),
            'erro': erro,
            'convergiu': convergiu
        }

        if convergiu:
            historico_passos.append(passo_info)
            # ADICIONADO: Retornar P_spec e Q_spec
            return V, theta, historico_passos, pvpq, pq_index, P_spec, Q_spec

        H, N, M, L = build_jacobian(V, theta, Ybus, pq_index, pv_index)
        J = np.block([[H, N], [M, L]])
        
        passo_info['J'] = J
        passo_info['H'] = H
        passo_info['N'] = N
        passo_info['M'] = M
        passo_info['L'] = L

        try:
            dx = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, mismatch, rcond=None)[0]

        n_ang = len(pvpq)
        dtheta = dx[:n_ang]
        dV = dx[n_ang:]
        
        passo_info['dtheta'] = dtheta
        passo_info['dV'] = dV

        for i, bus_idx in enumerate(pvpq):
            theta[bus_idx] += dtheta[i]
        for i, bus_idx in enumerate(pq_index):
            V[bus_idx] += dV[i]

        passo_info['V_prox'] = V.copy()
        passo_info['theta_prox'] = theta.copy()

        historico_passos.append(passo_info)

    return V, theta, historico_passos, pvpq, pq_index, P_spec, Q_spec
