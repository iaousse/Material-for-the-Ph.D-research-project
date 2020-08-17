#import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
#######Joreskog's formula
def jor(gamma, beta, phi, psi):
    """The function takes the matrices gamma,
    beta, phi, and psi as arguments and returns
    the implied covariance matrix using
    Joreskog's formula"""
    p = beta.shape[0]
    q = gamma.shape[1]
    R_1 = phi
    inv_ident_beta = inv(np.eye(p) - beta)
    R_2 = phi.dot(gamma.T).dot(inv_ident_beta.T)
    R_3 = R_2.T
    R_4 = inv_ident_beta.dot(gamma.dot(phi).dot(gamma.T)+psi).dot(inv_ident_beta.T)
    R_jor = np.bmat([[R_1, R_2], [R_3, R_4]])
    return R_jor
####### Finite Iterative Method
def fim(gamma, beta, phi, psi, S):
    """The function takes the matrices gamma, beta,
    beta, phi, psi, and S as arguments and 
    returns the impliedcovariance matrix
    using FIM"""
    p = beta.shape[0]
    q = gamma.shape[1]
    A = np.block([[gamma, beta]])
    def recur(j) :
        if j==0 :
                return phi
        else:
                a = A[j - 1, :q + j - 1].reshape(1, q+j-1)
                r1 = recur(j-1)
                r2 = a.dot(r1)
                r3 = np.array(S[j-1,])
                return np.block([[r1, r2.T],
                                 [r2, r3]])
    return recur(p)
## Use the functions in the comparison between the
## proposed methods
if __name__ == '__main__':
    # set the seed for reproducibility
    np.random.seed(5550)
    # create a vector which will contain the difference
    # in each time we generate the vector of the model
    # parameters (100 times)
    dist_jor_fim_1 = np.empty(100)
    # Generate a handred vector of model parameters
    # no constraint generation
    for i in range(100):
        a, b, c, phi_1, phi_2, phi_12, s_11, s_22 = np.random.random(8)
        psi_1, psi_2 = np.random.random(2)
        gamma = np.array([[a, b], [0, 0]])
        beta = np.array([[0, 0],
                         [c, 0]])
        S = np.array([[s_11], [s_22]])
        phi = np.array([[phi_1, phi_12],
                        [phi_12, phi_2]])
        psi = np.diag([psi_1, psi_2])
        jor_m = jor(gamma, beta, phi, psi)
        fim_m = fim(gamma, beta, phi, psi, S)
        
        dist_jor_fim_1[i]=0.5*np.trace((jor_m-fim_m)**2)
    # Small variances of disturbances as constraints
    dist_jor_fim_2 = np.empty(100)
    for i in range(100):
        a, b, c, phi_1, phi_2, phi_12, s_11, s_22 = np.random.random(8)
        psi_1, psi_2 = np.random.random(2) / 100
        gamma = np.array([[a, b], [0, 0]])
        beta = np.array([[0, 0],
                         [c, 0]])
        S = np.array([[s_11],[s_22]])
        phi= np.array([[phi_1, phi_12],
                       [phi_12, phi_2]])
        psi = np.diag([psi_1, psi_2])
        jor_m = jor(gamma, beta, phi, psi)
        fim_m = fim(gamma, beta, phi, psi, S)
        
        dist_jor_fim_2[i] = 0.5*np.trace((jor_m-fim_m)**2)
    # Constrained variances 
    dist_jor_fim_3 = np.empty(100)
    for i in range(100):
        a, b, c, phi_1, phi_2, phi_12, s_11, s_22 = np.random.random(8)
        psi_1 =s_11-(a**2*phi_1+b**2*phi_2+2*a*b*phi_12)
        psi_2 = s_22-(c**2*s_11)
        gamma = np.array([[a, b], [0, 0]])
        beta = np.array([[0, 0],
                         [c, 0]])
        S = np.array([[s_11], [s_22]])
        phi = np.array([[phi_1, phi_12],
                        [phi_12, phi_2]])
        psi = np.diag([psi_1, psi_2])
        jor_m = jor(gamma, beta, phi, psi)
        fim_m = fim(gamma, beta, phi, psi, S)
        
        dist_jor_fim_3[i] = 0.5*np.trace((jor_m-fim_m)**2)

    # Print the maximum of the differences
    # in each simulation
    print(dist_jor_fim_1.max(),
          dist_jor_fim_2.max(),
          dist_jor_fim_3.max())
    # Plot the differences
    index = np.arange(100)
    plt.plot(index, dist_jor_fim_1, 'r',
             index, dist_jor_fim_2, 'b--',
             index, dist_jor_fim_3, 'b')
    plt.xlabel('Index')
    plt.ylabel('Difference')
    plt.legend(['fisrt simulation',
                'second simulation',
                'third simulation'])
    plt.grid('on')
    plt.show()
