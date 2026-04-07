import numpy as np

def build_ybus(buses, lines):

    n = len(buses)

    Ybus = np.zeros((n,n), dtype=complex)

    for line in lines:

        i = line["from"]-1
        j = line["to"]-1

        R = line["R"]
        X = line["X"]
        Bsh = line["Bsh"]

        Z = complex(R,X)
        Y = 1/Z

        Ybus[i,i] += Y + complex(0,Bsh/2)
        Ybus[j,j] += Y + complex(0,Bsh/2)

        Ybus[i,j] -= Y
        Ybus[j,i] -= Y

    return Ybus