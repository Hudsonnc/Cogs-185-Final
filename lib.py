import tensorflow as tf
import numpy as np
EPS = 1e-8

####################Stein##########################################
def sq_dists(Z):
    A = tf.reshape(Z, (tf.shape(Z)[0], -1))
    r = tf.reduce_sum(A*A, 1)
    r = tf.reshape(r, [-1, 1])
    sqdists = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return sqdists

def med(D):
    m = tf.contrib.distributions.percentile(D, 50)
    return m
    
def kernel(Z, h = -1):
    sqdists = sq_dists(Z)
    if h <= 0:
        medsq = med(sqdists)
        h = medsq / tf.log(tf.to_float(tf.shape(Z)[0]))
    h = tf.stop_gradient(h)
    ker = tf.exp(-sqdists/h)
    A = tf.tile(tf.expand_dims(Z, 0), (tf.shape(Z)[0],1,1))
    T = tf.transpose(A, (1,0,2)) - A
    dker_dz = -2*tf.multiply(tf.tile(tf.expand_dims(ker, 2), (1,1,tf.shape(Z)[1])), T)/h
    return(ker, dker_dz)

def phi_star(Z, logp, h = -1):
    ker, dker = kernel(Z, h=h)
    dlogp = tf.gradients(logp(Z), Z)[0]
    phi_mat = tf.einsum('ij,ik->ijk', ker, dlogp) + dker
    phi_mean = tf.reduce_mean(phi_mat, 0)
    return(phi_mean)


##########################Density Functions#######################
def binary_max_ent(X, tau=.1):
    p = .5
    density = tau * tf.pow(p/(tf.pow(X, tau) + EPS) + (1-p)/(tf.pow(1-X, tau) + EPS), -2) * \
                     (p/(tf.pow(X, tau+1) + EPS) * (1-p)/(tf.pow(1-X, tau+1) + EPS))
    log_density = tf.log(density + EPS)
    log_density = tf.reduce_sum(log_density, -1)
    return(log_density)

###########################Gumbel########################################
def gumbel_soft(pi, g, tau=.1):
    logit = tf.log(pi + EPS)
    y = tf.nn.softmax((g+logit)/tau, axis = -1)
    return(y)