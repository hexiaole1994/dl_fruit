import os.path
import zipfile
import numpy as np
from PIL import Image

zip_dst = "archive.zip"
zip_dir = "fruit"
train_dir = "fruit/train/train/"
train_data = [
    ['Undefined', 0, '未定义'],
    ['Apple Braeburn', 1, '红苹果'],
    ['Apple Granny Smith', 2, '绿苹果'],
    ['Apricot', 3, '杏'],
    ['Avocado', 4, '鳄梨'],
    ['Banana', 5, '香蕉'],
    ['Blueberry', 6, '蓝莓'],
    ['Cactus fruit', 7, '仙人掌'],
    ['Cantaloupe', 8, '哈密瓜'],
    ['Cherry', 9, '樱桃'],
    ['Clementine', 10, '克莱门氏小柑橘'],
    ['Corn', 11, '玉米'],
    ['Cucumber Ripe', 12, '熟黄瓜'],
    ['Grape Blue', 13, '蓝葡萄'],
    ['Kiwi', 14, '猕猴桃'],
    ['Lemon', 15, '柠檬'],
    ['Limes', 16, '酸橙'],
    ['Mango', 17, '芒果'],
    ['Onion White', 18, '白洋葱'],
    ['Orange', 19, '橘子'],
    ['Papaya', 20, '木瓜'],
    ['Passion Fruit', 21, '百香果'],
    ['Peach', 22, '桃子'],
    ['Pear', 23, '梨'],
    ['Pepper Green', 24, '绿辣椒'],
    ['Pepper Red', 25, '红辣椒'],
    ['Pineapple', 26, '菠萝'],
    ['Plum', 27, '李子'],
    ['Pomegranate', 28, '石榴'],
    ['Potato Red', 29, '红土豆'],
    ['Raspberry', 30, '树莓'],
    ['Strawberry', 31, '草莓'],
    ['Tomato', 32, '西红柿'],
    ['Watermelon', 33, '西瓜'],
]

def img_show(arr):
    img = Image.fromarray(arr)
    img.show()

def normalize(img):
    arr = np.array(img)
    arr = arr.astype(np.float32)
    arr /= 255.0
    arr = arr.reshape(3, 100, 100)
    return arr

def load_fruit():
    x_train = []
    t_train = []
    x_test = []
    t_test = []
    if not os.path.exists(zip_dir):
        with zipfile.ZipFile(zip_dst, 'r') as f:
            print("unzip file " + zip_dst)
            f.extractall(zip_dir)
            print("unzip file " + zip_dst + " ok")
    for arr in train_data:
        dir = train_dir + arr[0]
        key = arr[1]
        if key == 0:
            continue
        files = os.listdir(dir)
        for file in files:
            img = Image.open(os.path.join(dir, file))
            arr = normalize(img)
            x_train.append(arr)
            t_train.append(key)
    shuf = np.random.permutation(len(x_train))
    x_train = np.array(x_train)
    t_train = np.array(t_train)
    x_test = np.array(x_test)
    t_test = np.array(t_test)
    x_train = x_train[shuf]
    t_train = t_train[shuf]
    sp = int(0.2 * len(x_train))
    x_test = x_train[:sp]
    t_test = t_train[:sp]
    x_train = x_train[sp:]
    t_train = t_train[sp:]
    return (x_train, t_train), (x_test, t_test)
