import numpy as np

import theano
import theano.tensor as T

import scipy.io as sio
import scipy.optimize as sop

### Parameters 

batch_size = 256 # number of img patches to train on
img_length = 12 # dimension of img patch
img_size = img_length**2 
oc = 1 # overcompleteness
num_neurons = oc * img_size
BUFF = 4
lamb = 0.001 # sparsity weight
eta = 1e-6 # learning rate
rank = 4 # number of cols of L matrix (how many nonzero eigenvalues do we want H to have?)

"""
Dendritic model: 
    x = A (x * H * x) + noise
where:
    x: image pixels, img_size X batch_size vector
    A: basis functions, img_size X num_neurons matrix
    H: weight tensor, num_neurons X img_length X img_length

Sparse coding:
    x = A s + noise
once again, x are image pixels, A are the basis functions; s is the activity vector

We can think of s = x * H * x as the activity vector of the dendritic model.
"""

### set up generative model parameters

# they need to be float32 in order to be compatible with GPU
x = T.fmatrix('x')
# basis functions
A = T.fmatrix('A')
# low rank approx for H tensor
# here, (H_i) = (L_i) * (L_i)^T
L = T.ftensor3('H')

### set up learning and cost functions

print "Creating functions...\n"

# add some negative eigenvalues
switch = np.resize(np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]), (num_neurons, rank, rank))
# activity vector
H = T.batched_tensordot(T.batched_tensordot(L, switch, axes = 1), L.dimshuffle(0,2,1), axes=1)
s = ((T.tensordot(H, x, axes=[[2], [0]]))*x.dimshuffle('x', 0, 1)).sum(axis = 1) # create activity vector from tensor
# estimate for image after normalizing A
A_norm = A/np.sqrt((A**2).sum(axis = 0, keepdims = True))
x_hat = T.dot(A_norm, s)  
# cost functions
recon_cost = 0.5*((x_hat - x)**2).sum(axis=0).mean()
sparsity_cost = abs(s).mean(axis = 0).mean()
total_cost = recon_cost + lamb * sparsity_cost
# gradients
grad_L, grad_A = T.grad(total_cost, [L,A])

### load image data

images = sio.loadmat('IMAGES.mat')
images = images['IMAGES']
img_length_x, img_length_y, num_img = images.shape

# function to retrieve a batch of image patches 
def get_img_patch():
    patches = np.zeros((img_size, batch_size))
    for j in xrange(batch_size):
        row = BUFF + int((img_length_x-img_length-2.*BUFF)*np.random.rand())
        col = BUFF + int((img_length_y-img_length-2.*BUFF)*np.random.rand())
        patches[:, j] = images[row:row+img_length,
                               col:col+img_length,
                               int(num_img * np.random.rand())].ravel()
    return patches

patches = get_img_patch()

### theano functions for interfacing with BFGS

cost_th = theano.function([A, L, x], total_cost, allow_input_downcast=True)
grad_A_th = theano.function([A, L, x], grad_A, allow_input_downcast=True)
grad_L_th = theano.function([A, L, x], grad_L, allow_input_downcast=True)
snr = theano.function([A, L, x], (x.var(axis = 0)/(x - x_hat).var(axis = 0)).mean())

### helper functions 

# takes in vector of entries for A,L; returns A and L
def split_AL(A_and_L):
    a, l = np.split(A_and_L, [img_size * num_neurons])
    a = a.reshape(img_size, num_neurons)
    l = l.reshape(num_neurons, img_size, rank)
    return a,l

# takes in matrix A, tensor L; returns a ravelled vector containing both
def merge_AL(A, L):
    return np.append(A.flatten(), L.flatten())

### cost and gradient functions BFGS will interface directly with

def cost_func(A_and_L):
    a, l = split_AL(A_and_L)
    return cost_th(a, l, patches).astype('float64')

def grad_func(A_and_L):
    a, l = split_AL(A_and_L)
    grad_A = grad_A_th(a, l, patches)
    grad_L = grad_L_th(a, l, patches)
    return (np.append(grad_A.flatten(), grad_L.flatten())).astype('float64')

A_guess = np.random.randn(img_size, num_neurons)
A_guess = A_guess/np.sqrt((A_guess**2).sum(axis = 0, keepdims = True))
L_guess = np.random.randn(num_neurons, img_size, rank)
L_guess = L_guess/np.sqrt((L_guess**2).sum(axis=(1,2), keepdims=True))
guess = merge_AL(A_guess, L_guess)

print "Starting up BFGS...\n"

A_and_L, cost_min, dic = sop.fmin_l_bfgs_b(cost_func, guess, fprime = grad_func, iprint = 50, factr = 0, maxfun = 50000, maxiter = 50000)

np.savetxt('results-3.txt', A_and_L, delimiter=',')
