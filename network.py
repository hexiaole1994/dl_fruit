import pickle
import matplotlib.pyplot as plt
from layers import *
from functions import *
from collections import OrderedDict

class network:
    def __init__(self,
                 input_dim=(1, 28, 28),
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100,
                 output_size=10,
                 weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size-filter_size+2*filter_pad) // filter_stride + 1
        pool_output_size = filter_num * (conv_output_size//2) * (conv_output_size//2)
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        self.layers = OrderedDict()
        self.layers['Conv1'] = convolution_layer(self.params['W1'], self.params['b1'], stride=filter_stride, pad=filter_pad)
        self.layers['Relu1'] = relu_layer()
        self.layers['Pool1'] = pooling_layer(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = affine_layer(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = relu_layer()
        self.layers['Affine2'] = affine_layer(self.params['W3'], self.params['b3'])
        self.lastlayer = softmax_with_loss_layer()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        x = self.predict(x)
        y = self.lastlayer.forward(x, t)
        return y

    def accuracy(self, x, t, batch_size=50):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range((x.shape[0]-1)//batch_size+1):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def showacc(self, train_acc, test_acc):
        x = np.arange(len(train_acc))
        plt.plot(x, train_acc, label='train acc')
        plt.plot(x, test_acc, label='test acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

    def numerical_gradient(self, x, t):
        f = lambda w: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(f, self.params['W1'])
        grads['b1'] = numerical_gradient(f, self.params['b1'])
        grads['W2'] = numerical_gradient(f, self.params['W2'])
        grads['b2'] = numerical_gradient(f, self.params['b2'])
        grads['W3'] = numerical_gradient(f, self.params['W3'])
        grads['b3'] = numerical_gradient(f, self.params['b3'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        return grads

    def gradient_check(self, img_batch, label_batch):
        grads_nf = self.numerical_gradient(img_batch, label_batch)
        grads_bp = self.gradient(img_batch, label_batch)
        for key in grads_nf.keys():
            diff = np.average(np.abs(grads_nf[key] - grads_bp[key]))
            print(key + ": " + str(diff))

    def save_params(self, fname='params.pkl'):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(fname, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, fname='params.pkl'):
        with open(fname, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        self.layers['Conv1'].W = params['W1']
        self.layers['Conv1'].b = params['b1']
        self.layers['Affine1'].W = params['W2']
        self.layers['Affine1'].b = params['b2']
        self.layers['Affine2'].W = params['W3']
        self.layers['Affine2'].b = params['b3']