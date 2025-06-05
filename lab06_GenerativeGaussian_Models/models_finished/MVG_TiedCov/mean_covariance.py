import numpy as np

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C


def compute_C(D, mu):
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return C