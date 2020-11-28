"""The following program is capable of computing the
covariance matrix implied by a recusrive SEM model
with observed variables in four different methods"""

################## Import the necessary dependencies
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import inv

################### Defin the function for the
################### of the covariance matrix
def compute_sigma(a_star, phi_star, sample_cov, method):
    """ This function takes three matrices and a key
    argument :the matrices that define the
    structure of the model a_star and phi_star,
    the sample covariance matrix S, and the method
    which define the type of the method to used.
    The output is the implied covariance matrix"""
    # Define the necessary variables
    p = a_star.shape[0]
    q = a_star.shape[1] - 2*p
    phi = phi_star[:q, :q]
    gamma = a_star[:, :q]
    beta = a_star[:, q+p:]
    psi = phi_star[q:, q:]
    A = np.block([gamma, beta])
    g_extract = np.block([[np.eye(q), \
                           np.zeros((q, p)), \
                           np.zeros((q, p))],
                          [np.zeros((p, q)), \
                           np.zeros((p, p)), \
                           np.eye(p)]])
    # The Joreskog's formula
    if method == "jor":
        i_b_inv = inv(np.eye(p)-beta)
        r_1 = phi
        r_2 = phi.dot(gamma.T.dot(i_b_inv.T))
        r_3 = i_b_inv.dot((gamma.dot(phi.\
                                     dot(gamma.T)))+\
                          psi).dot(i_b_inv.T)
        return np.block([[r_1, r_2],
                         [r_2.T, r_3]])
    # The Finite Iterative Method
    elif method == "fim":
        def recur(j):
            if j == 0:
                return phi
            else:
                a = A[j - 1, :q + j - 1]. \
                    reshape(1, q + j - 1)
                r1 = recur(j - 1)
                r2 = a.dot(r1)
                r3 = sample_cov[q + j - 1, q + j -1]
                return np.block([[r1, r2.T],
                                 [r2, r3]])
        return recur(p)
    # The first proposed method
    elif method == "new_fim1":
        def recur(j):
            if j == 0:
                return phi_star
            else:
                a = a_star[j - 1, :q + p + j - 1].\
                    reshape(1, p+q+j-1)
                r1 = recur(j-1)
                r2 = a.dot(r1)
                r3 = sample_cov[q + j - 1, q + j -1]
                return np.block([[r1, r2.T],
                                 [r2, r3]])
        return g_extract.dot(recur(p)).dot(g_extract.T)
    # The second proposed method
    elif method == "new_fim2":
        def recur(j):
            if j == 0:
                return phi_star
            else:
                a = a_star[j - 1, :q + p + j - 1].\
                    reshape(1, p+q+j-1)
                r1 = recur(j-1)
                r2 = a.dot(r1)
                r3 = r2.dot(a.T)
                return np.block([[r1, r2.T],
                                 [r2, r3]])
        return g_extract.dot(recur(p)).dot(g_extract.T)
    # In case the method was entered wrongly
    else:
        print("Please enter one of the following methods:\
               'fim', 'jor', 'new_fim1', 'new_fim2'")
#################### The support functions
# The function to_theta
def to_theta(a_star, phi_star):
    """Thhis function takes a_star and phi_star
    as arguments and return the vector of free
    model parameters"""
    theta_0 = np.hstack([a_star.flatten(), phi_star.\
                         flatten()])
    index = np.logical_and(theta_0 != 0, theta_0 != 1)
    theta = theta_0[index]
    return theta, index
# The function to_matrix
def to_matrix(theta, a_star, phi_star):
    """ this function takes some vector theta, a_star,
    and phi_star as arguments and returns the new a_star
    and phi_star where the places of free parameters are
    replaced by the new values in theta"""
    p = a_star.shape[0]
    q = a_star.shape[1] - 2 * p
    theta_1 = np.hstack([a_star.flatten(), \
                         phi_star.flatten()])
    theta_1[to_theta(a_star, phi_star)[1]]= theta
    a_star_res = theta_1[:p*(q+2*p)].reshape(p, q+2*p)
    phi_star_res =  theta_1[p*(q+2*p):].reshape(q+p,q+p)
    for i in range(q+p):
        for j in range(q+p):
            if i<j:
                phi_star_res[j,i]=phi_star_res[i,j]
    return a_star_res, phi_star_res
#################### The fit function
def fit_func(theta, a_star, phi_star,sample_cov,  \
             method="jor", fit="uls"):
    """ This function takes theta, a_star, phi_star, S,
    and the method and returns the ULS function"""
    cov2 = sample_cov
    numb_endo = a_star.shape[0]
    a_star_new, phi_star_new = to_matrix(theta, \
                                         a_star, phi_star)
    if method == "jor":
        cov1 = compute_sigma(\
            a_star_new, phi_star_new, sample_cov, "jor")
        if fit == "uls":
            return .5*np.trace(np.dot((cov1-cov2).T, \
                                      cov1-cov2))
        elif fit == "ml":
            return np.log(np.linalg.det(cov1)) + \
                   np.trace(cov2.dot(np.linalg.\
                                     inv(cov1)))\
                                     -np.log(\
                                         np.linalg.\
                                         det(cov2)) \
                                         - numb_endo
        elif fit == "gls":
            return .5*np.trace((cov1.\
                                dot(np.linalg.\
                                    inv(cov2))-\
                                np.eye(cov1.\
                                       shape[0]))**2)
        else:
            print("Enter the fit function correctly")
    elif method == "fim":
        cov1 = compute_sigma(a_star_new, phi_star_new, \
                             sample_cov, "fim")
        if fit == "uls":
            return .5*np.trace(np.dot((cov1-cov2).T, \
                                      cov1-cov2))
        elif fit == "ml":
            return np.log(np.linalg.det(cov1)) + \
                   np.trace(cov2.dot(inv(cov1))) \
                   - np.log(np.linalg.det(cov2)) \
                   - numb_endo
        elif fit == "gls":
            return .5*np.trace((cov1.dot(inv(cov2))\
                                -np.eye(cov1.shape[0]))**2)
        else:
            print("Enter the fit function correctly")
    elif method == "new_fim1":
        cov1 = compute_sigma(a_star_new, phi_star_new, \
                             sample_cov, "new_fim1")
        if fit == "uls":
            return .5*np.trace(np.dot((cov1-cov2).T,\
                                      cov1-cov2))
        elif fit == "ml":
            return np.log(np.linalg.det(cov1)) + \
                   np.trace(cov2.dot(np.linalg.\
                                     inv(cov1))) \
                   - np.log(np.linalg.det(cov2)) \
                   - numb_endo
        elif fit == "gls":
            return .5*np.trace((cov1.dot(\
                inv(cov2))-\
                                np.eye(cov1.shape[0]))**2)
        else:
            print("Enter the fit function correctly")
    elif method == "new_fim2":
        cov1 = compute_sigma(a_star_new, phi_star_new, \
                             sample_cov, "new_fim2")
        if fit == "uls":
            return .5*np.trace(np.dot((cov1-cov2).T, \
                                      cov1-cov2))
        elif fit == "ml":
            return np.log(np.linalg.det(cov1)) + \
                   np.trace(cov2.dot(np.linalg.\
                                     inv(cov1)))\
                   -np.log(np.linalg.det(cov2)) \
                   - numb_endo
        elif fit == "gls":
            return .5*np.trace((cov1.dot(np.linalg.\
                                         inv(cov2))-\
                                np.eye(cov1.\
                                       shape[0]))**2)
        else:
            print("Enter the fit function correctly")
    else:
        print("Enter the method correctly")
###################### The optimazation function
def optim(fit_func, a_star, phi_star, sample_cov, \
          optimizer='BFGS', method='fim', fit='uls'):
    """This function takes fit_func, a_star, phi_star
    , sample covariance, Algorithm of optimization,
    the method to use, and the type fit('uls' for now)
    The function returns the argument of the minimum
    of the fit function
    """
    a_star_new = a_star.copy()
    phi_star_new = phi_star.copy()
    theta = to_theta(a_star_new, phi_star_new)[0]
    solution = minimize(fit_func, theta, args=\
                        (a_star,phi_star,\
                         sample_cov,\
                         method, "uls"),
                        method = optimizer,
                        jac=None,
                        hess=None,
                        hessp=None,
                        bounds=None,
                        constraints=(),
                        tol=None,
                        callback=None,
                        options=None)
    return solution.x
############ Computation of the GFI function
def compute_gfi (fit_func, a_star, phi_star, sample_cov, \
                 optimizer='BFGS', method='fim', \
                 fit='uls'):
    """This function takes fit_func, a_star, phi_star, the
    sample covariance, the algorithm of optimization,
    the method and the type of fit function ('uls' for now)
    and returns the GFI of the model"""
    resid_matrix = sample_cov - compute_sigma(\
        *to_matrix(optim(fit_func, a_star, phi_star,\
                         sample_cov, optimizer, method, \
                         fit), a_star, phi_star), \
        sample_cov, method)
    return np.trace(resid_matrix.dot(resid_matrix.T)) \
           / (sample_cov ** 2).sum()
