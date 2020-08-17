import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from program import *
np.random.seed(5551)
dist_jor_newfim2_a1 = np.empty(100)
dist_fim_newfim1_a1 = np.empty(100)
dist_newfim1_newfim2_a1 = np.empty(100)
dist_jor_newfim2_b1 = np.empty(100)
dist_fim_newfim1_b1 = np.empty(100)
dist_newfim1_newfim2_b1 = np.empty(100)
for i in range(100):
    a, b, phi_1, psi_1, psi_2, rho, s_22, s_33 = np.random.uniform(0, 1, 8)
    a_star = np.array([[a, 1, 0, 0, 0],
                       [0, 0, 1, b, 0]])
    phi_star_a= np.array([[phi_1, 0, 0],
                          [0, psi_1, 0],
                          [0, 0, psi_2]])
    phi_star_b = np.array([[phi_1, 0, rho],
                           [0, psi_1, 0],
                           [rho, 0, psi_2]])
    S = np.array([[0, 0, 0],
                  [0, s_22, 0],
                  [0, 0, s_33]])

    jor_ma = compute_sigma(a_star, phi_star_a,
                           S, 'jor')
    jor_mb = compute_sigma(a_star, phi_star_b,
                           S, 'jor')
    fim_ma = compute_sigma(a_star, phi_star_a,
                           S, 'fim')
    fim_mb = compute_sigma(a_star, phi_star_b,
                           S, 'fim')
    newfim1_ma = compute_sigma(a_star,
                               phi_star_a,
                               S, 'new_fim1')
    newfim1_mb = compute_sigma(a_star,
                               phi_star_b,
                               S, 'new_fim1')
    newfim2_ma = compute_sigma(a_star, phi_star_a,
                               S, 'new_fim2')
    newfim2_mb = compute_sigma(a_star, phi_star_b,
                               S, 'new_fim2')
    dist_jor_newfim2_a1[i] = 0.5 * np.trace((jor_ma-newfim2_ma).dot((jor_ma - newfim2_ma).T))
    dist_fim_newfim1_a1[i] = 0.5 * np.trace((fim_ma - newfim1_ma).dot((fim_ma -newfim1_ma).T))
    dist_newfim1_newfim2_a1[i] = 0.5 * np.trace((newfim1_ma - newfim2_ma).dot((newfim1_ma - newfim2_ma).T))
    dist_jor_newfim2_b1[i] = 0.5 * np.trace((jor_mb - newfim2_mb).dot((jor_mb - newfim2_mb).T) )
    dist_fim_newfim1_b1[i] = 0.5 * np.trace((fim_mb - newfim1_mb).dot((fim_mb - newfim1_mb).T))
    dist_newfim1_newfim2_b1[i] = 0.5 * np.trace((newfim1_mb - newfim2_mb).dot((newfim1_mb - newfim2_mb).T))
############### small error variances
    dist_jor_newfim2_a2 = np.empty(100)
    dist_fim_newfim1_a2 = np.empty(100)
    dist_newfim1_newfim2_a2 = np.empty(100)
    dist_jor_newfim2_b2 = np.empty(100)
    dist_fim_newfim1_b2 = np.empty(100)
    dist_newfim1_newfim2_b2 = np.empty(100)
    for i in range(100):
        a, b, phi_1, s_22, s_33 = np.random.uniform(0, 1, 5)
        psi_1, psi_2, rho = np.random.random(3)/1000
        a_star = np.array([[a, 1, 0, 0, 0],
                           [0, 0, 1, b, 0]])
        phi_star_a = np.array([[phi_1, 0, 0],
                               [0, psi_1, 0],
                               [0, 0, psi_2]])
        phi_star_b = np.array([[phi_1, 0, rho],
                               [0, psi_1, 0],
                               [rho, 0, psi_2]])
        S = np.array([[0, 0, 0],
                      [0, s_22, 0],
                      [0, 0, s_33]])
        jor_ma = compute_sigma(a_star, phi_star_a,
                               S, 'jor')
        jor_mb = compute_sigma(a_star, phi_star_b,
                               S, 'jor')
        fim_ma = compute_sigma(a_star, phi_star_a,
                               S, 'fim')
        fim_mb = compute_sigma(a_star, phi_star_b,
                               S, 'fim')
        newfim1_ma = compute_sigma(a_star, phi_star_a,
                                   S, 'new_fim1')
        newfim1_mb = compute_sigma(a_star, phi_star_b,
                                   S, 'new_fim1')
        newfim2_ma = compute_sigma(a_star, phi_star_a,
                                   S, 'new_fim2')
        newfim2_mb = compute_sigma(a_star, phi_star_b,
                                   S, 'new_fim2')
        dist_jor_newfim2_a2[i] = 0.5 * np.trace((jor_ma - newfim2_ma).dot((jor_ma - newfim2_ma).T))
        dist_fim_newfim1_a2[i] = 0.5 * np.trace((fim_ma - newfim1_ma).dot((fim_ma - newfim1_ma).T))
        dist_newfim1_newfim2_a2[i] = 0.5 * np.trace((newfim1_ma - newfim2_ma).dot((newfim1_ma - newfim2_ma).T))
        dist_jor_newfim2_b2[i] = 0.5 * np.trace((jor_mb - newfim2_mb).dot((jor_mb - newfim2_mb).T))
        dist_fim_newfim1_b2[i] = 0.5 * np.trace((fim_mb - newfim1_mb).dot((fim_mb - newfim1_mb).T))
        dist_newfim1_newfim2_b2[i] = 0.5 * np.trace((newfim1_mb - newfim2_mb).dot((newfim1_mb - newfim2_mb).T))

    ############### constrained error variances
    dist_jor_newfim2_a3 = np.empty(100)
    dist_fim_newfim1_a3 = np.empty(100)
    dist_newfim1_newfim2_a3 = np.empty(100)
    dist_jor_newfim2_b3 = np.empty(100)
    dist_fim_newfim1_b3 = np.empty(100)
    dist_newfim1_newfim2_b3 = np.empty(100)
    for i in range(100):
        a, b, phi_1, s_22, s_33, rho = np.random.uniform(0, 1, 6)
        psi_1 = s_22-a**2*phi_1
        psi_2 = s_33-a**2*b**2*phi_1-b**2*psi_1-2*a*b*rho
        a_star = np.array([[a, 1, 0, 0, 0],
                           [0, 0, 1, b, 0]])
        phi_star_a = np.array([[phi_1, 0, 0],
                               [0, psi_1, 0],
                               [0, 0, psi_2]])
        phi_star_b = np.array([[phi_1, 0, rho],
                               [0, psi_1, 0],
                               [rho, 0, psi_2]])
        S = np.array([[0, 0, 0],
                      [0, s_22, 0],
                      [0, 0, s_33]])
        jor_ma = compute_sigma(a_star, phi_star_a,
                               S, 'jor')
        jor_mb = compute_sigma(a_star, phi_star_b,
                               S, 'jor')
        fim_ma = compute_sigma(a_star, phi_star_a,
                               S, 'fim')
        fim_mb = compute_sigma(a_star, phi_star_b,
                               S, 'fim')
        newfim1_ma = compute_sigma(a_star, phi_star_a,
                                   S, 'new_fim1')
        newfim1_mb = compute_sigma(a_star, phi_star_b,
                                   S, 'new_fim1')
        newfim2_ma = compute_sigma(a_star, phi_star_a,
                                   S, 'new_fim2')
        newfim2_mb = compute_sigma(a_star, phi_star_b,
                                   S, 'new_fim2')
        dist_jor_newfim2_a3[i] = 0.5 * np.trace((jor_ma - newfim2_ma).dot((jor_ma - newfim2_ma).T))
        dist_fim_newfim1_a3[i] = 0.5 * np.trace((fim_ma - newfim1_ma).dot((fim_ma - newfim1_ma).T))
        dist_newfim1_newfim2_a3[i] = 0.5 * np.trace((newfim1_ma - newfim2_ma).dot((newfim1_ma - newfim2_ma).T))
        dist_jor_newfim2_b3[i] = 0.5 * np.trace((jor_mb - newfim2_mb).dot((jor_mb - newfim2_mb).T))
        dist_fim_newfim1_b3[i] = 0.5 * np.trace((fim_mb - newfim1_mb).dot((fim_mb - newfim1_mb).T))
        dist_newfim1_newfim2_b3[i] = 0.5 * np.trace((newfim1_mb - newfim2_mb).dot((newfim1_mb - newfim2_mb).T))
index = np.arange(100)
plt.subplot(321)
plt.plot(index, dist_fim_newfim1_a1,'r',
         index, dist_jor_newfim2_a1, 'b:',
         index, dist_newfim1_newfim2_a1, 'y--')
plt.xlabel('Index')
plt.ylabel('Difference')
plt.title('First simulation model a')
plt.grid('on')
plt.subplot(322)
plt.plot(index, dist_fim_newfim1_b1, 'r',
         index, dist_jor_newfim2_b1, 'b:',
         index, dist_newfim1_newfim2_b1, 'y--')
plt.xlabel('Index')
plt.ylabel('Difference')
plt.title('First simulation model b')
####
plt.grid('on')
plt.subplot(323)
plt.plot(index, dist_fim_newfim1_a2, 'r',
         index, dist_jor_newfim2_a2, 'b:',
         index, dist_newfim1_newfim2_a2, 'y--')
plt.xlabel('Index')
plt.ylabel('Difference')
plt.title('Second simulation model a')
plt.grid('on')
plt.subplot(324)
plt.plot(index, dist_fim_newfim1_b2, 'r',
         index, dist_jor_newfim2_b2, 'b:',
         index, dist_newfim1_newfim2_b2, 'y--')
plt.xlabel('Index')
plt.ylabel('Difference')
plt.gca().legend([r'$\Delta_1$', r'$\Delta_2$',
                  r'$\Delta_3$'],
                 loc='center left',
                 bbox_to_anchor=(1, 0.5))
plt.title('Second simulation model b')
plt.grid('on')
######
plt.subplot(325)
plt.plot(index, dist_fim_newfim1_a3, 'r',
         index, dist_jor_newfim2_a3, 'b:',
         index, dist_newfim1_newfim2_a3, 'y--')
plt.xlabel('Index')
plt.ylabel('Difference')
plt.title('Third simulation model a')
plt.grid('on')
plt.subplot(326)
plt.plot(index, dist_fim_newfim1_b3, 'r',
         index, dist_jor_newfim2_b3, 'b:',
         index, dist_newfim1_newfim2_b3, 'y--')
plt.xlabel('Index')
plt.ylabel('Difference')
plt.title('Third simulation model b')
plt.grid('on')
plt.tight_layout()
plt.show()
