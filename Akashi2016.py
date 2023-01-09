import numpy as np
import cv2


image_path = 'Please enter the image path'

I = cv2.imread(image_path, cv2.IMREAD_COLOR)
n_row, n_col, n_ch = I.shape
M = 3
N = n_row * n_col
R = 10
I = I.reshape(N,M).conjugate().transpose(1,0)
i_s = np.ones((3,1), dtype=np.uint8)/np.sqrt(3)
H = 254 * np.random.uniform(0,1,(R,N)) + 1
W_d = 254 * np.random.uniform(0,1,(3,R-1)) + 1

W_d = W_d / np.linalg.norm(W_d, 2, axis=0)

W = np.concatenate([i_s, W_d], axis=1)

A = np.ones(M).astype(np.uint8)
lamb = 3
eps = np.exp(-15)
F_t_1 = np.inf

i = 0
max_iter = 10000
while True:
    print('iter : %5d'%i)
    W_bar = W / np.linalg.norm(W, 2, axis=0)

    H = H * (W_bar.conjugate().transpose(1,0)@I) / \
            (W_bar.conjugate().transpose(1,0)@W_bar@H+lamb)
    H_d = H[1:,:]
    
    Vl = I-(i_s@H[0:1,:])
    Vl = np.where(Vl > 0, Vl, 0)

    W_d_bar = W_bar[:,1:]
    
    W_d = W_d_bar * (Vl @ H_d.conjugate().transpose(1,0) + \
        W_d_bar * (A @ W_d_bar @ H_d @ H_d.conjugate().transpose(1,0))) / \
        (W_d_bar @ H_d @ H_d.conjugate().transpose(1,0) + \
        W_d_bar * (A @ Vl @ H_d.conjugate().transpose(1,0)))

    W = np.concatenate([i_s, W_d], axis=1)
    
    F_t = 0.5 * np.linalg.norm((I- W @ H), 'fro') + lamb * np.sum(H[:])

    if (np.abs(F_t-F_t_1) < eps * np.abs(F_t)) or (i >= max_iter):
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

#I_s = np.clip(I_s, 0,1)*255
I_d = np.clip(I_d, 0,1)*255

I = cv2.imread(image_path, cv2.IMREAD_COLOR)
I_s = I-I_d

cv2.imwrite('python_specular.png', I_s)
cv2.imwrite('python_diffuse.png', I_d)
