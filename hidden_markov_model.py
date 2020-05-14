#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:17:51 2020

@author: ted
"""
# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# Discrete Hidden Markov Model (HMM) with scaling
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
import csv


def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


class HMM:
    def __init__(self, M):
        self.M = M # number of hidden states
        
        self.V = 1
        self.pi = np.ones(self.M) / self.M # initial state distribution
        self.A = random_normalized(self.M, self.M) # state transition matrix
        self.B = random_normalized(self.M, self.V) # output distribution
    
    def fit(self, X, max_iter=30):
        np.random.seed(123)
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        # determine V, the vocabulary size
        # assume observables are already integers from 0..V-1
        # X is a jagged array of observed sequences
        V = max(max(x) for x in X) + 1
        N = len(X)

        self.pi = np.ones(self.M) / self.M # initial state distribution
        self.A = random_normalized(self.M, self.M) # state transition matrix
        self.B = random_normalized(self.M, V) # output distribution

#        print("initial A:", self.A)
#        print("initial B:", self.B)

        costs = []
        print('M: ',self.M)
        for it in range(max_iter):
            print("it:", it)
            # alpha1 = np.zeros((N, self.M))
            alphas = []
            betas = []
            scales = []
            logP = np.zeros(N)
            for n in range(N):
                x = X[n]
                T = len(x)
                scale = np.zeros(T)
                # alpha1[n] = self.pi*self.B[:,x[0]]
                alpha = np.zeros((T, self.M))
                alpha[0] = self.pi*self.B[:,x[0]]
                scale[0] = alpha[0].sum()
                alpha[0] /= scale[0]
                for t in range(1, T):
                    alpha_t_prime = alpha[t-1].dot(self.A) * self.B[:, x[t]]
                    scale[t] = alpha_t_prime.sum()
                    alpha[t] = alpha_t_prime / scale[t]
                logP[n] = np.log(scale).sum()
                alphas.append(alpha)
                scales.append(scale)

                beta = np.zeros((T, self.M))
                beta[-1] = 1
                for t in range(T - 2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t+1]] * beta[t+1]) / scale[t+1]
                betas.append(beta)


            cost = np.sum(logP)
            costs.append(cost)

            # now re-estimate pi, A, B
            self.pi = np.sum((alphas[n][0] * betas[n][0]) for n in range(N)) / N

            den1 = np.zeros((self.M, 1))
            den2 = np.zeros((self.M, 1))
            a_num = np.zeros((self.M, self.M))
            b_num = np.zeros((self.M, V))
            for n in range(N):
                x = X[n]
                T = len(x)
                den1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T
                den2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T

                # numerator for A
                # a_num_n = np.zeros((self.M, self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num[i,j] += alphas[n][t,i] * betas[n][t+1,j] * self.A[i,j] * self.B[j, x[t+1]] / scales[n][t+1]
                # a_num += a_num_n

                # numerator for B
                # for i in range(self.M):
                #     for j in range(V):
                #         for t in range(T):
                #             if x[t] == j:
                #                 b_num[i,j] += alphas[n][t][i] * betas[n][t][i]
                for i in range(self.M):
                    for t in range(T):
                        b_num[i,x[t]] += alphas[n][t,i] * betas[n][t,i]
            self.A = a_num / den1
            self.B = b_num / den2

        
#        print("A:", self.A)
#        print("B:", self.B)
#ms        print("pi:", self.pi)

        return self.A
        return self.B
        return self.pi
        return self.V
        
        plt.plot(costs)
        return plt.show()

    def log_likelihood(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x)
        scale = np.zeros(T)
        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi*self.B[:,x[0]]
        print(alpha[0])
        scale[0] = alpha[0].sum()
        print(scale[0])
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha_t_prime = alpha[t-1].dot(self.A) * self.B[:, x[t]]
            scale[t] = alpha_t_prime.sum()
            alpha[t] = alpha_t_prime / scale[t]
        return np.log(scale).sum()

    def log_likelihood_multi(self, X):
        return np.array([self.log_likelihood(x) for x in X])

    def get_state_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = np.log(self.pi) + np.log(self.B[:,x[0]])
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(self.B[j, x[t]])
                psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states


def import_data_set(file_name_string,set_list):
    X = [None]*len(set_list)

    for w in range(len(set_list)):
        element_list = set_list[w]
        file_name = '{file}{set_l}.csv'.format(file = file_name_string,set_l = element_list)
#        print(w)
        with open(file_name, 'r') as f:  #opens PW file
            reader = csv.reader(f)
            X[w] = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
        for i in range(len(X[w])):
            for j in range(len(X[w][i])):           
                X[w][i][j] = int(X[w][i][j])
    total = []
    for i in X:
        total +=i
    X = total
    return X

def fit_bond_credit(data_set,m,max_iter):
    
    X = data_set
    
#    n_hidden_states = [3,5,10,15,20,25,30,35]
#    results = dict(zip(n_hidden_states,[0]*len(n_hidden_states)))
#    for i in n_hidden_states:
    hmm = HMM(m)
    hmm.fit(X,max_iter)
    L = hmm.log_likelihood_multi(X).sum()
    print("LL with fitted params:", L)

    # try true values
#    hmm.pi = np.array([0.5, 0.5])
#    hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
#    hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])
#    L = hmm.log_likelihood_multi(X).sum()
#    print("LL with true params:", L)

    # try viterbi
    print("Best state sequence for:", X[0])
    print(hmm.get_state_sequence(X[0]))
    return hmm
    


#if __name__ == '__main__':
 #   fit_bond_credit()
 
ms_starting = [2,4,8,16,32,64]
test_performance_starting = [None]*6
test_performance_2 = [None]*5
test_performance_3 = [None]*5
test_performance_cv_1 = [None]*5
test_performance_cv_2 = [None]*5
test_performance_cv_3 = [None]*5
test_performance_cv_4 = [None]*5



print('M Starting')
test_set = import_data_set('hmm_fr_data_set_',[0])
training_set = import_data_set('hmm_fr_data_set_',[1,2,3,4])

for i in ms_starting:
    training_set_model = fit_bond_credit(training_set,i,6)
    test_performance_starting[ms_starting.index(i)] = training_set_model.log_likelihood_multi(test_set).sum()
best_performance =  ms_starting[max(range(len(test_performance_starting)), key=test_performance_starting.__getitem__)]

plt.plot(ms_starting,test_performance_starting)
plt.savefig('M_vs_Likelihood_Test_1.png')
plt.show()


ms_2 = [int(round(best_performance*.6)),int(round(best_performance*.8)),int(round(best_performance)),int(round(best_performance*1.2)),int(round(best_performance*1.4))]

print('M 2')
for i in ms_2:
    training_set_model = fit_bond_credit(training_set,i,10)
    test_performance_2[ms_2.index(i)] = training_set_model.log_likelihood_multi(test_set).sum()
best_performance_2 =  ms_2[max(range(len(test_performance_2)), key=test_performance_2.__getitem__)]

plt.plot(ms_2,test_performance_2)
plt.savefig('M_vs_Likelihood_Test_2.png')
plt.show()


with open("test_performance_2.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(map(lambda x: [x], test_performance_2))
    
with open("ms_2.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(map(lambda x: [x], ms_2))
    
    
print('M 3')

ms_3 = [int(round(best_performance_2*.75)),int(round(best_performance_2*.9)),int(round(best_performance_2)),int(round(best_performance_2*1.1)),int(round(best_performance_2*1.25))]

for i in ms_3:
    training_set_model = fit_bond_credit(training_set,i,15)
    test_performance_3[ms_3.index(i)] = training_set_model.log_likelihood_multi(test_set).sum()
best_performance_3 =  ms_3[max(range(len(test_performance_3)), key=test_performance_3.__getitem__)]

plt.plot(ms_3,test_performance_3)
plt.savefig('M_vs_Likelihood_Test_3.png')
plt.show()

with open("test_performance_3.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(map(lambda x: [x], test_performance_3))
    
with open("ms_3.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(map(lambda x: [x], ms_3))

plt.plot(ms_3,test_performance_3)
plt.savefig('Cross_validation_test_0.png')
plt.show()

print('CV 1')


test_set = import_data_set('hmm_fr_data_set_',[1])
training_set = import_data_set('hmm_fr_data_set_',[0,2,3,4])
for i in ms_3:
    training_set_model = fit_bond_credit(training_set,i,15)
    test_performance_cv_1[ms_3.index(i)] = training_set_model.log_likelihood_multi(test_set).sum()
best_performance_cv_1 =  ms_3[max(range(len(test_performance_cv_1)), key=test_performance_cv_1.__getitem__)]

plt.plot(ms_3,test_performance_cv_1)
plt.savefig('Cross_validation_test_1.png')
plt.show()

with open("test_performance_cv_1.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(map(lambda x: [x], test_performance_cv_1))
    

print('CV 2')


test_set = import_data_set('hmm_fr_data_set_',[2])
training_set = import_data_set('hmm_fr_data_set_',[0,1,3,4])
for i in ms_3:
    training_set_model = fit_bond_credit(training_set,i,15)
    test_performance_cv_2[ms_3.index(i)] = training_set_model.log_likelihood_multi(test_set).sum()
best_performance_cv_2 =  ms_3[max(range(len(test_performance_cv_2)), key=test_performance_cv_2.__getitem__)]

plt.plot(ms_3,test_performance_cv_2)
plt.savefig('Cross_validation_test_2.png')
plt.show()

with open("test_performance_cv_2.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(map(lambda x: [x], test_performance_cv_2))

print('CV 3')


test_set = import_data_set('hmm_fr_data_set_',[3])
training_set = import_data_set('hmm_fr_data_set_',[0,1,2,4])
for i in ms_3:
    training_set_model = fit_bond_credit(training_set,i,15)
    test_performance_cv_3[ms_3.index(i)] = training_set_model.log_likelihood_multi(test_set).sum()
best_performance_cv_3 =  ms_3[max(range(len(test_performance_cv_3)), key=test_performance_cv_3.__getitem__)]

plt.plot(ms_3,test_performance_cv_3)
plt.savefig('Cross_validation_test_3.png')
plt.show()

with open("test_performance_cv_3.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(map(lambda x: [x], test_performance_cv_3))

print('CV 4')


test_set = import_data_set('hmm_fr_data_set_',[4])
training_set = import_data_set('hmm_fr_data_set_',[0,1,2,3])
for i in ms_3:
    training_set_model = fit_bond_credit(training_set,i,15)
    test_performance_cv_4[ms_3.index(i)] = training_set_model.log_likelihood_multi(test_set).sum()
best_performance_cv_4 =  ms_3[max(range(len(test_performance_cv_4)), key=test_performance_cv_4.__getitem__)]

plt.plot(ms_3,test_performance_cv_4)
plt.savefig('Cross_validation_test_4.png')
plt.show()

with open("test_performance_cv_4.csv", "w",newline="") as f:
    writer = csv.writer(f)
    writer.writerows(map(lambda x: [x], test_performance_cv_4))

np.savetxt('hmm_pi.csv', training_set_model.pi, delimiter=",",fmt = '%.3f')
np.savetxt('hmm_A.csv', training_set_model.A, delimiter=",",fmt = '%.3f')
np.savetxt('hmm_B.csv', training_set_model.B, delimiter=",",fmt = '%.3f')


'''
Number of observable states in each of the 5 folds, and the union of their complement.
Also test to see if the initial observable states in the test set are present as initial observalbe states in the training set.
'''

dt = import_data_set('hmm_fr_data_set_',[0,1,2,3,4])
dt_0 = import_data_set('hmm_fr_data_set_',[0])
dt_1 = import_data_set('hmm_fr_data_set_',[1])
dt_2 = import_data_set('hmm_fr_data_set_',[2])
dt_3 = import_data_set('hmm_fr_data_set_',[3])
dt_4 = import_data_set('hmm_fr_data_set_',[4])
dt_0_0 = import_data_set('hmm_fr_data_set_',[1,2,3,4])
dt_1_0 = import_data_set('hmm_fr_data_set_',[0,2,3,4])
dt_2_0 = import_data_set('hmm_fr_data_set_',[0,1,3,4])
dt_3_0 = import_data_set('hmm_fr_data_set_',[0,1,2,4])
dt_4_0 = import_data_set('hmm_fr_data_set_',[0,1,2,3])

datasets = [dt,dt_0,dt_1,dt_2,dt_3,dt_4,dt_0_0,dt_1_0,dt_2_0,dt_3_0,dt_4_0]
datasets_s = ['dt','dt_0','dt_1','dt_2','dt_3','dt_4','dt_0_0','dt_1_0','dt_2_0','dt_3_0','dt_4_0']

val = [None]*11
for d in datasets:
    firsts = [None]
    for seq in d:
        firsts.append(seq[0])
    firsts = set(firsts)
    firsts = list(firsts)
    val[datasets.index(d)] = firsts
    
if set(val[1]).issubset(val[6]) == False:
    print('Data set 0 is NOT test-able')
else:
    print('Data set 0 is test-able')

if set(val[2]).issubset(val[7]) == False:
    print('Data set 1 is NOT test-able')
else:
    print('Data set 1 is test-able')

if set(val[3]).issubset(val[8]) == False:
    print('Data set 2 is NOT test-able')
else:
    print('Data set 2 is test-able')
    
if set(val[4]).issubset(val[9]) == False:
    print('Data set 3 is NOT test-able')
else:
    print('Data set 3 is test-able')

if set(val[5]).issubset(val[10]) == False:
    print('Data set 4 is NOT test-able')
else:
    print('Data set 4 is test-able')
    

    