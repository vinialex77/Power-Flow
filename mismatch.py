import numpy as np

def calc_power(V, theta, Ybus):

    n = len(V)

    P = np.zeros(n)
    Q = np.zeros(n)

    G = Ybus.real
    B = Ybus.imag

    for i in range(n):

        for k in range(n):

            P[i] += V[i]*V[k]*(
                G[i,k]*np.cos(theta[i]-theta[k]) +
                B[i,k]*np.sin(theta[i]-theta[k])
            )

            Q[i] += V[i]*V[k]*(
                G[i,k]*np.sin(theta[i]-theta[k]) -
                B[i,k]*np.cos(theta[i]-theta[k])
            )

    return P,Q