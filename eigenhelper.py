import numpy as np

def characteristic_polynomial(A): #this only works with 2x2 matrices
    A = np.array(A)
    print("this is the A you gave me. make sure its right:\n{}".format(A))
    a = float(A[0][0])
    b = float(A[0][1])
    c = float(A[1][0])
    d = float(A[1][1])
    squared_value = 1.0
    singlex_value = -a + -d
    number = (a*d)-(b*c)
    print("x^2 + {}x + {}".format(singlex_value, number))
    ary = [squared_value, singlex_value, number]
    solns = np.roots(ary)
    print("solutions (eigenvalues) are:\n {}".format(solns))

def convert_to_eigenbasis(C,D,exponent):
    # T^n = C @ D^n @ C^(-1)
    # where C is a matrix of the eigenvectors of T    
    # D is the diagonal matrix of eigenvalues
    #Dn = None
    #for n in range(exponent):
    #    n = n+1
    #    if n==1:
    #        Dn = D
    #    Dn = D @ D
    Dn = np.linalg.matrix_power(D, exponent)
    Cneg1 = np.linalg.inv(C)
    Cn = C @ Dn @ Cneg1
    return Cn

def page_rank_helper_no_damping(L, iters):
    n = L.shape[0]
    r = 100 * np.ones(n) / n
    for i in range(1, iters):
        r = L @ r
    return r
    
def page_rank_helper(L, damping_factor): #used to take r and a general number of iterations, but using this np.linalg.norm thing we can get it to auto-iterate until its done
    n = L.shape[0]
    L = damping_factor * L + (1-damping_factor)/n * np.ones([n,n])
    r = 100 * np.ones(n) / n
    prevR = r    
    r = L @ r
    i = 0
    #for i in range(1, iterations):
    #    r = d*(L @ r) + (1-d)/n
    while np.linalg.norm(prevR - r) > 0.01 : #I'm still not 100% sure what norm does but its evaluating the degree of difference between the prevR and r
        prevR = r
        r = L @ r
        i += 1
    print(str(i) + " iterations to convergence")
    return r