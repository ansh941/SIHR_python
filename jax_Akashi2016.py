import numpy as np
import cv2

import jax
from jax import jit, random, device_put
import jax.numpy as jnp

import time

@jit
def separate_ds(W,I,H,lamb,i_s):
    W_bar = W / jnp.linalg.norm(W, 2, axis=0)

    H = H * (W_bar.conjugate().transpose(1,0)@I) / \
            (W_bar.conjugate().transpose(1,0)@W_bar@H+lamb)

    H_d = H[1:,:]
    
    Vl = I-(i_s@H[:1,:])
    Vl = jnp.where(Vl > 0, Vl, 0)

    W_d_bar = W_bar[:,1:]
    
    W_d = W_d_bar * (Vl @ H_d.conjugate().transpose(1,0) + \
        W_d_bar * (A @ W_d_bar @ H_d @ H_d.conjugate().transpose(1,0))) / \
        (W_d_bar @ H_d @ H_d.conjugate().transpose(1,0) + \
        W_d_bar * (A @ Vl @ H_d.conjugate().transpose(1,0)))

    W = jnp.concatenate([i_s, W_d], axis=1)
    
    F_t = 0.5 * jnp.linalg.norm((I-W@H),'fro') + lamb * jnp.sum(H[:])

    return W, H, F_t
    
key = random.PRNGKey(0)

image_path = 'Please enter the image path'
I = cv2.imread(image_path, cv2.IMREAD_COLOR)
n_row, n_col, n_ch = I.shape
M = 3
N = n_row * n_col
R = 7
I = I.reshape(N,M).conjugate().transpose(1,0)

i_s = jnp.ones((3,1), dtype=np.uint8)/jnp.sqrt(3)

H = 254 * random.uniform(key, minval=0, maxval=1, shape=(R,N)) + 1
W_d = 254 * random.uniform(key, minval=0, maxval=1, shape=(3,R-1)) + 1

W_d = W_d / jnp.linalg.norm(W_d, 2, axis=0)
W_d = device_put(W_d)

W = jnp.concatenate([i_s, W_d], axis=1)
W = device_put(W)

A = jnp.ones(M).astype(np.uint8)
A = device_put(A)
lamb = 3
eps = jnp.exp(-18)
F_t_1 = jnp.inf

i = 0
max_iter = 10000
start = time.time()
while True:
    print('iter : %5d'%i)
    W, H, F_t = separate_ds(W,I,H,lamb,i_s)
    if (jnp.abs(F_t-F_t_1) < eps * jnp.abs(F_t)) or (i >= max_iter):
        break
    
    F_t_1 = F_t
    i += 1
W_d = W[:, 1:]

#H_s = H[:1,:]
H_d = H[1:,:]

#I_s = i_s @ H_s
I_d = W_d @ H_d

#I_s = I_s.conjugate().transpose(1,0).reshape(n_row, n_col, n_ch) / 255
I_d = I_d.conjugate().transpose(1,0).reshape(n_row, n_col, n_ch) / 255

#I_s = jnp.clip(I_s, 0,1)*255
I_d = jnp.clip(I_d, 0,1)*255
print(time.time()-start)

cpus = jax.devices('cpu')

I = I.conjugate().transpose(1,0).reshape(n_row, n_col, n_ch)
I_s = np.asarray(device_put(I-I_d, cpus[0]))
I_d = np.asarray(device_put(I_d, cpus[0]))

cv2.imwrite('python_specular_2016.png', I_s)
cv2.imwrite('python_diffuse_2016.png', I_d)

