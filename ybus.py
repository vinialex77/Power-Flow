import numpy as np

def build_ybus(buses, lines):
    n = len(buses)
    Ybus = np.zeros((n,n), dtype=complex)

    # 1. Contribuição das Linhas (Série + Shunt da Linha)
    for line in lines:
        i = line["from"]-1
        j = line["to"]-1
        Z = complex(line["R"], line["X"])
        Y = 1/Z
        Bsh_L = line["Bsh"]

        Ybus[i,i] += Y + complex(0, Bsh_L/2)
        Ybus[j,j] += Y + complex(0, Bsh_L/2)
        Ybus[i,j] -= Y
        Ybus[j,i] -= Y

    # 2. Contribuição do Shunt da Barra (Capacitor/Indutor de Barra)
    for idx, b in enumerate(buses):
        # b["Bsh_bus"] virá da interface
        Bsh_B = b.get("Bsh_bus", 0.0)
        Ybus[idx, idx] += complex(0, Bsh_B)

    return Ybus
