import numpy as np

import theano
import theano.tensor as T

import scipy.io as sio
import scipy.optimize as sop

print "Setting up..."

batch_size = 10000 # number of img patches to train on
img_length = 12 # dimension of img patch
img_size = img_length**2 
oc = 1 # overcompleteness
num_neurons = oc * img_size
BUFF = 4
lamb = 0.001 # sparsity weight
eta = 1e-6 # learning rate

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
# weight tensor
H = T.ftensor3('H')

### set up learning and cost functions

# activity vector
s = ((T.tensordot(H, x, axes=[[2], [0]]))*x.dimshuffle('x', 0, 1)).sum(axis = 1) # create activity vector from tensor
# estimate for image
x_hat = T.dot(A, s)  
# cost functions
recon_cost = 0.5*((x_hat - x)**2).sum(axis=0).mean()
sparsity_cost = abs(s).mean(axis = 0).mean()
total_cost = recon_cost + lamb * sparsity_cost
# gradients
grad_H, grad_A = T.grad(total_cost, [H,A])

### load image data

print "Loading images..."
images = sio.loadmat('IMAGES.mat')
images = images['IMAGES']
img_length_x, img_length_y, num_img = images.shape

# function to retrieve a batch of image patches 
def get_img_patch():
    for j in xrange(batch_size):
        row = BUFF + int((img_length_x-img_length-2.*BUFF)*np.random.rand())
        col = BUFF + int((img_length_y-img_length-2.*BUFF)*np.random.rand())
        patches[:, j] = images[row:row+img_length,
                               col:col+img_length,
                               int(num_img * np.random.rand())].ravel()
    return patches

### some pre-processing for sorting out the entries of H. H should be symmetric in the last two indices

# obtain the upper indices for triangular matrix
triupp_ind = np.triu_indices(img_size)
triu_length = (triupp_ind[0].shape)[0]
# indices for upper triangle matrix for each entry of H (since is it a 3-index tensor)
H_indices = (np.repeat(np.arange(num_neurons), triu_length), 
            np.tile(triupp_ind[0], num_neurons),
            np.tile(triupp_ind[1], num_neurons))
# indices for diagonal matrix for each entry of H
H_di_indices = (np.repeat(np.arange(num_neurons), img_size), np.tile(np.arange(img_size),num_neurons), np.tile(np.arange(img_size),num_neurons))

### helper functions 

# takes in vector of entries for A,H; returns A and H
def split_AH(A_and_H):
    a = A_and_H[:(img_size * num_neurons)].reshape(img_size, num_neurons)
    h = A_and_H[(img_size * num_neurons):].reshape(num_neurons, img_size, img_size)
    return a,h

# takes in A, H; returns a one-dimensional vector containing all entries
def merge_AH(A,H):
    return np.append(A.flatten(), H.flatten())

# takes in a vector of entries for A,H; returns a vector the removes the redundant symmetric entries in H
def split_AH_sym(A_and_H):
    a, h = np.split(A_and_H, [(img_size * num_neurons)])
    a = a.reshape(img_size, num_neurons)
    h = make_sym_H(h)
    return a, h

# takes in A, (symmetric) H; returns a vector containing the entries of A and the non-redundant entries of H
def merge_AH_sym(A,H):
    return np.append(A.flatten(), H[H_indices])

# take in a vector of H entries and convert this to a 3-index tensor symmetric in the last two entries.
def make_sym_H(H_vec):
    h = np.zeros((num_neurons, img_size, img_size))
    h[H_indices] = H_vec
    h = h + np.transpose(h, axes=(0,2,1))
    h[H_di_indices] /= 2.
    return h

# take in a (symmetric) tensor H and return its entries collapsed into a vector 
def make_H_vec(H):
    return H[H_indices]
    
### theano functions for interfacing with BFGS

cost_th = theano.function([A, H, x], total_cost, allow_input_downcast=True)
grad_A_th = theano.function([A, H, x], grad_A, allow_input_downcast=True)
grad_H_th = theano.function([A, H, x], grad_H, allow_input_downcast=True)
snr = theano.function([A, H, x], (x.var(axis = 0)/(x - x_hat).var(axis = 0)).mean())

### python functions that return the cost and gradients; BFGS will optimize on cost_func

def cost_func(A_and_H):
    a, h = split_AH_sym(A_and_H)
    return cost_th(a, h, patches).astype('float64')

def grad_func(A_and_H):
    a, h = split_AH_sym(A_and_H)
    grad_A = grad_A_th(a, h, patches)
    grad_H = make_H_vec(grad_H_th(a, h, patches))
    return (np.append(grad_A.flatten(), grad_H)).astype('float64')
    
### begin BFGS algorithm to optimize on the cost 

print "Starting up BFGS..."
# set up initial guesses
A_guess = np.random.randn(img_size, num_neurons)
A_guess = A_guess/np.sqrt((A_guess**2).sum(axis = 0, keepdims = True))
H_guess = np.random.randn(num_neurons, img_size, img_size)
H_guess = H_guess + np.transpose(H_guess,axes=(0,2,1)) # make symmetric
H_guess = H_guess/np.sqrt((H_guess**2).sum(axis = (1,2), keepdims = True)) # normalize the matrix for each neuron
guess = merge_AH_sym(A_guess, H_guess).astype('float64')

# get a set of image patches
patches = get_img_patch()

A_and_H, cost_min, dic = sop.fmin_l_bfgs_b(cost_func, guess, fprime = grad_func, iprint = 100, maxiter = 30000)


np.savetxt('results_sym.txt', A_and_H, delimiter=',')


