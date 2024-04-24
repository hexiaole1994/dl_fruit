import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x -= np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        y = y.T
        return(y)
    x -= np.max(x)
    y = np.exp(x) / np.sum(np.exp(x))
    return y
def cross_entropy_error(x, t):
    delta = 1e-7
    if x.ndim == 1:
        t = t.reshape(1, t.size)
        x = x.reshape(1, x.size)
    if t.size == x.size:
        t = t.argmax(axis=1)
    batch_size = x.shape[0]
    y = - np.sum(np.log(x[np.arange(batch_size), t]+delta)) / batch_size
    return y

def numerical_gradient(f, x):
    h = 1e-4
    grads = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        save = x[i]
        x[i] = save - h
        y1 = f(x)
        x[i] = save + h
        y2 = f(x)
        grads[i] = (y2-y1) / (2*h)
        x[i] = save
        it.iternext()
    return grads

def im2col(x, FH, FW, stride=1, pad=0):
    N, C, H, W = x.shape
    out_h = (H-FH+2*pad) // stride + 1
    out_w = (W-FW+2*pad) // stride + 1
    img = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    col = np.zeros((N, C, FH, FW, out_h, out_w))
    for y in range(FH):
        y_max = y + stride * out_h
        for x in range(FW):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, FH, FW, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H-FH+2*pad) // stride + 1
    out_w = (W-FW+2*pad) // stride + 1
    col = col.reshape(N, out_h, out_w, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H+2*pad+stride-1, W+2*pad+stride-1))
    for y in range(FH):
        y_max = y + stride * out_h
        for x in range(FW):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H+pad, pad:W+pad]
