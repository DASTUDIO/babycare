from keras import models, layers
import jieba
import numpy as np
import care.models as db

models_data_dir = './models_data/data'

d = open('./dic/dic.txt').readlines()

word_dic = {}


def load_map():  # init
    for i in range(len(d)):
        res = d[i].split(' ')
        word_dic[res[0]] = res[1]


def _do_split(words: str):
    source = jieba.cut(words)
    str_res = '<split/>'.join(source)
    l = str_res.split('<split/>')
    res = []
    for i in range(len(l)):
        res.append(l[i])
    return res


def load_data(file_path: str):  # return ndarray
    with open(file_path) as b:
        lines = b.readlines()
        r = np.zeros((len(lines), 585380))
        for i in range(len(lines)):
            words = _do_split(lines[i]) # 一行的分词
            for j in range(len(words)):
                if words[j] in word_dic:
                    r[i][int(word_dic[words[j]])] = 1.
        return r


# def load_db_data():  # return ndarray
#     lines = db.data.objects.all()
#     x = np.zeros((len(lines), 585380))
#     y = np.zeros(len(lines))
#     for i in range(len(lines)):
#         y = lines[i].res
#         words = _do_split(lines[i].text) # 一行的分词
#         for j in range(len(words)):
#             if words[j] in word_dic:
#                 x[i][int(word_dic[words[j]])] = 1.
#     return x, y


def gen_train(string1:str,string2:str):  # return ndarray
    res = np.zeros((2, 585380))
    words1 = _do_split(string1)
    for j in range(len(words1)):
        if words1[j] in word_dic:
            res[0][int(word_dic[words1[j]])] = 1.
    words2 = _do_split(string2)
    for j in range(len(words2)):
        if words2[j] in word_dic:
            res[1][int(word_dic[words2[j]])] = 1.

    return res


load_map()

bad_x = load_data('./data_set/bad_data_train')
bad_y = np.ones(len(bad_x))
good_x = load_data('./data_set/good_data_train')
good_y = np.zeros(len(good_x))
train_x = np.append(bad_x, good_x, axis=0)
train_y = np.append(bad_y, good_y, axis=0).astype('float32')

# train_x, train_y = load_db_data()

try:
    m = models.load_model(models_data_dir)
except:
    m = models.Sequential()

m.add(layers.Dense(16, activation='relu'))
m.add(layers.Dense(16, activation='relu'))
m.add(layers.Dense(1, activation='sigmoid'))
m.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
m.fit(train_x, train_y, epochs=20, batch_size=20)
m.save(models_data_dir)
