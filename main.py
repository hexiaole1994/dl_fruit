import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from fruit import *
from mnist import *
from network import *
from trainer import *

def open_file():
    r = tk.Tk()
    r.withdraw()
    fpath = filedialog.askopenfilename()
    return fpath

(x_train, t_train), (x_test, t_test) = load_fruit()
#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=False)
max_epochs = 10
nw = network(input_dim=(3, 100, 100),
                       conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                       hidden_size=50, output_size=34,
                       weight_init_std=0.01)
t = trainer(nw,
            x_train, t_train, x_test, t_test,
            epochs=max_epochs,
            min_batch_size=100,
            optimizer='adam', optimizer_param={'lr': 0.001})
t.train()
while 1:
    fpath = open_file()
    if not fpath:
        break
    img = Image.open(fpath)
    x = normalize(img)
    x = x.reshape(1, 3, 100, 100)
    y = nw.predict(x)
    print(fpath + " is " + train_data[np.argmax(y)][2])
