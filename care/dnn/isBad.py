from keras import models
import jieba
import numpy as np
import tensorflow as tf

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

m = models.load_model(models_data_dir)

graph = tf.get_default_graph()

def predict(string:str):
    global graph
    with graph.as_default():
        return float(m.predict(gen_train(string, '初生者的心态是禅宗的做法，这让你能看到事物的本质.'))[0])




