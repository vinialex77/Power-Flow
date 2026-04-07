import numpy as np


def build_jacobian(V, theta, Ybus, pq_index, pv_index):

    G = Ybus.real
    B = Ybus.imag

    n = len(V)

    pvpq = pv_index + pq_index

    npvpq = len(pvpq)
    npq = len(pq_index)

    H = np.zeros((npvpq, npvpq))
    N = np.zeros((npvpq, npq))
    M = np.zeros((npq, npvpq))
    L = np.zeros((npq, npq))


    # H = dP/dθ
    for a,i in enumerate(pvpq):

        for b,k in enumerate(pvpq):

            if i == k:

                soma = 0

                for m in range(n):

                    soma += V[i]*V[m]*(
                        -G[i,m]*np.sin(theta[i]-theta[m])
                        + B[i,m]*np.cos(theta[i]-theta[m])
                    )

                H[a,b] = soma - V[i]**2 * B[i,i]

            else:

                H[a,b] = V[i]*V[k]*(
                    G[i,k]*np.sin(theta[i]-theta[k])
                    - B[i,k]*np.cos(theta[i]-theta[k])
                )


    # N = dP/dV
    for a,i in enumerate(pvpq):

        for b,k in enumerate(pq_index):

            if i == k:

                soma = 0

                for m in range(n):

                    soma += V[m]*(
                        G[i,m]*np.cos(theta[i]-theta[m])
                        + B[i,m]*np.sin(theta[i]-theta[m])
                    )

                N[a,b] = soma + G[i,i]*V[i]

            else:

                N[a,b] = V[i]*(
                    G[i,k]*np.cos(theta[i]-theta[k])
                    + B[i,k]*np.sin(theta[i]-theta[k])
                )


    # M = dQ/dθ
    for a,i in enumerate(pq_index):

        for b,k in enumerate(pvpq):

            if i == k:

                soma = 0

                for m in range(n):

                    soma += V[i]*V[m]*(
                        G[i,m]*np.cos(theta[i]-theta[m])
                        + B[i,m]*np.sin(theta[i]-theta[m])
                    )

                M[a,b] = -soma + V[i]**2 * G[i,i]

            else:

                M[a,b] = -V[i]*V[k]*(
                    G[i,k]*np.cos(theta[i]-theta[k])
                    + B[i,k]*np.sin(theta[i]-theta[k])
                )


    # L = dQ/dV
    for a,i in enumerate(pq_index):

        for b,k in enumerate(pq_index):

            if i == k:

                soma = 0

                for m in range(n):

                    soma += V[m]*(
                        G[i,m]*np.sin(theta[i]-theta[m])
                        - B[i,m]*np.cos(theta[i]-theta[m])
                    )

                L[a,b] = soma - B[i,i]*V[i]

            else:

                L[a,b] = V[i]*(
                    G[i,k]*np.sin(theta[i]-theta[k])
                    - B[i,k]*np.cos(theta[i]-theta[k])
                )

    return H,N,M,L