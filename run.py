#!python3
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

import numpy as np

D = np.matrix([
        [0, 20, 50, 100, 150, 200, 100, 150],
        [0, 0,  40, 20,  30,  50,  20,  25],
        [0, 0,  0,  100, 150, 200, 100, 85],
        [0, 0,  0,  0,   40,  30,  150, 89],
        [0, 0,  0,  0,   0,   80,  70,  70],
        [0, 0,  0,  0,   0,   0,   60,  60],
        [0, 0,  0,  0,   0,   0,   0,   80],
        [0, 0,  0,  0,   0,   0,   0,   0]
    ])
D = D + D.transpose()

print("Cost matrix:")
print(D)
print("\n\n")

#consume vector
C = np.array([50, 100, 200, 100, 100, 30, 0, 10])
#produce vector
P = np.array([100, 200, 100, 10, 20, 100, 50, 10])

print("Consume vector", C)
print("Produce vector", P)
print("SUM(C) - SUM(P) = ", np.sum(C), " - ", np.sum(P), " = ", np.sum(C - P))
print("We got 0, so it means that we deal with closed transport problem\n\n")

PdC = P-C
CP = PdC[PdC >= 0]      #Cities-producent
CC = -PdC[PdC < 0]      #Cities-consuments
print("Produce vector - Consume vector = ", PdC)
print("Axis with negative values are cities-consuments")

D = D[PdC >=0][:, PdC < 0]

print("\n\nCol: Consument x Row: Producent \n", D, "\n\n")

# Building matrix of pseudcosts
PC = D.copy()
PC_MinRow = np.array(PC.min(axis=1)).flatten()
PC = PC - np.dot(np.ones(PC.shape).transpose(), np.diag(PC_MinRow)).transpose()
print(PC, "\n\n")
PC_MinCol = np.array(PC.min(axis=0)).flatten()
PC = PC - np.dot(np.ones(PC.shape), np.diag(PC_MinCol))
print("Matrix of pseudocosts: \n", PC, "\n\n")

print("As we can see the last city can produce and want consume the same amount of prodcuts, so we can remove the last row in pseudocosts matrix. As well, dont forget that car can transport products in amount of 5 tons. This means that we can divide each value in matrix by 5\n")
PC = PC[0:-1,:] / 5
CP = CP[:-1] / 5
CC = CC / 5
print("So we get the following pseudocosts matrix:\n", PC, "\n")
print("As well, devide by 5 values of producents and consuments: ")
print("Cities producents: \n", CP.reshape((-1,1)))
print("Cities consuments: ", CC, "\n\n")

X = np.matrix([[10,0,0], [10,4,6], [0,14,0], [0,0,10]])
X = np.matrix([[10,0,0], [10,5,5], [0,13,1], [0,0,10]])
print("As we can see this is acceptable solution: \n", X, "\n\n")

def isAcceptableSolution(X, CP, CC):
    if np.any(X < 0):
        return False

    if np.any( CP - np.array(X.sum(axis=0)).flatten() <= 0 ):
        return False

    if not np.all( CC - np.array(X.sum(axis=1)).flatten() == 0 ):
        return False

    return True

def find_minimum_X(X, f, CP, CC, excludeX = np.array([])):
    #X = row:prod x col:cons
    #f function
    #CP = production cities abilities
    #CC = consumption cities requirements
    
    y = f(X)
    K = []

    for i1 in range(X.shape[0] - 1):
        for j1 in range(X.shape[1] - 1):
            for i2 in range(i1+1, X.shape[0]):
                for j2 in range(j1+1, X.shape[1]):
                    X1 = X.copy()
                    X2 = X.copy()

                    X1[i1,j1] -= 1
                    X1[i2,j2] -= 1
                    X1[i1,j2] += 1
                    X1[i2,j1] += 1
                    
                    X2[i1,j1] += 1
                    X2[i2,j2] += 1
                    X2[i1,j2] -= 1
                    X2[i2,j1] -= 1
                    
                    if np.all(X1 >= 0):
                        y1 = f(X1)
                        print(y1, y)
                        if y1 <= y:
                            K.append([i1, j1, i2, j2, 1, y1])

                    if np.all(X2 >= 0): 
                        y2 = f(X2)
                        print(y2, y)
                        if y2 <= y:
                            K.append([i1, j1, i2, j2, -1, y2])


    if len(K) == 0:
        return [X]

    K = np.array(K)
    yK = [k[5] for k in K]
    bestK = K[yK == np.min(yK)]
    acceptableX = []

    print("K: \n", K)
    print("bestK: \n", bestK)

    for i1, j1, i2, j2, t, yk in bestK:
        Xk = X.copy()
        
        I = np.zeros_like(Xk)
        I[i1,j1] -= 1
        I[i2,j2] -= 1
        I[i1,j2] += 1
        I[i2,j1] += 1
        I = t * I

        k = 1

        while True:
            if not isAcceptableSolution(Xk + (k + 1) * I, CP, CC):
                break

            k += 1

        acceptableX.append(Xk + k * I)
    
    acceptableX = np.array(acceptableX)

    if excludeX.size > 0:
        acceptableX = acceptableX[[np.any([np.any(Xp == Xk) for Xp in excludeX]) for Xk in acceptableX]]

    yOfAcceptableX = [f(Xk) for Xk in acceptableX]
    yMinOfAcceptableX = np.min(yOfAcceptableX)

    print("For main X is f(X) = ", y)
    for i in range(len(yOfAcceptableX)):
        print("f(X) = ", yOfAcceptableX[i], "for X = \n", acceptableX[i])
    
    if yMinOfAcceptableX < y:
        bestX = acceptableX[np.array(yOfAcceptableX) == yMinOfAcceptableX][0]
        
        return find_minimum_X(bestX, f, CP, CC)
    else:       # yMin == y
        possibleBestX = acceptableX[np.array(yOfAcceptableX) == yMinOfAcceptableX][0]
        newBestX = np.array([])

        for Xk in possibleBestX:
            break
            
            


f = lambda X: np.sum(np.multiply(X, PC))
bestX = find_minimum_X(X, f, CP, CC)

print("The best X for f(X) are below. For all of them they get minimal value of f(X) = ", f(bestX[0]) , "\n", bestX)

print("\n\n\n\n\n")
