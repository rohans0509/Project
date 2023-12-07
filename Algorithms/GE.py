import numpy as np

def GE(A,end_gf=False):
    n = len(A)
    A = np.array(A)

    growth_factors=[]

    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1
        for j in range(i + 1):
            U[j, i] = A[j, i] - np.dot(L[j, :j], U[:j, i])
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

        iteration_growth_factor = np.max(np.abs(U)) / np.max(np.abs(A))

        growth_factors.append(iteration_growth_factor)

    if end_gf:
        return L, U, growth_factors[-1]

    return L, U, max(growth_factors)