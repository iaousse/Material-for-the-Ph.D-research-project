import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from program import *
##The input arguments for the model
# The sample covariance (correlation) matrix
S = np.array([[1, -.304, -.323, -.083, -.132],
              [-.304, 1, .479, .263, .275],
              [-.323, .479, 1, .288, .489],
              [-.083, .263, .288, 1, .281],
              [-.132, .275, .489, .281, 1]])
# The matrix of structural parameters
a_star = np.array([[.5, .5, 1, 0, 0, 0, 0, 0],
                   [.5, .5, 0, 1, 0, .5, 0, 0],
                   [0, 0, 0, 0, 1, .5, .5, 0]])
# The matrix of the covariances of the
# augmented exogenous variables
phi_star = np.eye(5) * .5
phi_star[0, 1] = phi_star[1, 0] = .5
phi_star_b = np.eye(5) * .5
phi_star_b[0, 1] = phi_star[1, 0] = .5
phi_star_b[2, 3] = phi_star[3, 2] = .5
phi_star_b[0, 4] = phi_star[4, 0] = .5
phi_star_b[1, 4] = phi_star[4, 1] = .5
# model a :
theta_fim = optim(fit_func, a_star, phi_star,
                  S, optimizer='BFGS',
                  method='fim', fit='uls')
theta_new_fim1 = optim(fit_func, a_star,
                       phi_star, S, optimizer='BFGS',
                       method='new_fim1',
                       fit='uls')
theta_jor = optim(fit_func, a_star,
                  phi_star, S, optimizer='BFGS',
                  method='jor', fit='uls')
theta_new_fim2 = optim(fit_func, a_star,
                       phi_star, S, optimizer='BFGS',
                       method='new_fim2', fit='uls')
print(compute_gfi (fit_func, a_star, phi_star, S,
                   optimizer='BFGS', method='fim',
                   fit='uls'))
print(compute_gfi (fit_func, a_star, phi_star, S,
                   optimizer='BFGS', method='new_fim1',
                   fit='uls'))
print(compute_gfi (fit_func, a_star, phi_star, S,
                   optimizer='BFGS', method='jor',
                   fit='uls'))
print(compute_gfi (fit_func, a_star, phi_star, S,
                   optimizer='BFGS', method='new_fim2',
                   fit='uls'))
np.savetxt("a_a_star_fim.csv", to_matrix(theta_fim,
                                         a_star,
                                         phi_star)[0])
np.savetxt("a_phi_star_fim.csv", to_matrix(theta_fim,
                                           a_star,
                                           phi_star)[1])
np.savetxt("a_a_star_new_fim1.csv", to_matrix(theta_new_fim1,
                                              a_star,
                                              phi_star)[0])
np.savetxt("a_phi_star_new_fim1.csv", to_matrix(theta_new_fim1,
                                                a_star,
                                                phi_star)[1])
np.savetxt("a_a_star_new_fim2.csv", to_matrix(theta_new_fim2,
                                              a_star,
                                              phi_star)[0])
np.savetxt("a_phi_star_new_fim2.csv", to_matrix(theta_new_fim2,
                                                a_star,
                                                phi_star)[1])
np.savetxt("a_a_star_jor.csv", to_matrix(theta_jor,
                                         a_star,
                                         phi_star)[0])
np.savetxt("a_phi_star_jor.csv", to_matrix(theta_jor,
                                           a_star,
                                           phi_star)[1])

# model b :
theta_fim = optim(fit_func, a_star, phi_star_b, S,
                  optimizer='BFGS', method='fim',
                  fit='uls')
theta_new_fim1 = optim(fit_func, a_star, phi_star_b,
                       S, optimizer='BFGS', method='new_fim1',
                       fit='uls')
theta_jor = optim(fit_func, a_star, phi_star_b, S,
                  optimizer='BFGS', method='jor',
                  fit='uls')
theta_new_fim2 = optim(fit_func, a_star, phi_star_b, S,
                       optimizer='BFGS', method='new_fim2',
                       fit='uls')
print(compute_gfi (fit_func, a_star, phi_star_b, S,
                   optimizer='BFGS', method='fim', fit='uls'))
print(compute_gfi (fit_func, a_star, phi_star_b, S,
                   optimizer='BFGS', method='new_fim1',
                   fit='uls'))
print(compute_gfi (fit_func, a_star, phi_star_b, S,
                   optimizer='BFGS', method='jor',
                   fit='uls'))
print(compute_gfi (fit_func, a_star, phi_star_b, S,
                   optimizer='BFGS', method='new_fim2',
                   fit='uls'))
np.savetxt("b_a_star_fim.csv", to_matrix(theta_fim, a_star,
                                         phi_star_b)[0])
np.savetxt("b_phi_star_fim.csv", to_matrix(theta_fim, a_star,
                                           phi_star_b)[1])
np.savetxt("b_a_star_new_fim1.csv", to_matrix(theta_new_fim1,
                                              a_star,
                                              phi_star_b)[0])
np.savetxt("b_phi_star_new_fim1.csv", to_matrix(theta_new_fim1,
                                                a_star,
                                                phi_star_b)[1])
np.savetxt("b_a_star_new_fim2.csv", to_matrix(theta_new_fim2,
                                              a_star,
                                              phi_star_b)[0])
np.savetxt("b_phi_star_new_fim2.csv", to_matrix(theta_new_fim2,
                                                a_star,
                                                phi_star_b)[1])
np.savetxt("b_a_star_jor.csv", to_matrix(theta_jor,
                                         a_star,
                                         phi_star_b)[0])
np.savetxt("b_phi_star_jor.csv", to_matrix(theta_jor,
                                           a_star,
                                           phi_star_b)[1])
