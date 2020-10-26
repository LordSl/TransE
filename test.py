import json
import math
import random
import copy
import sys
from TransE import *
import numpy as np
import time

def file_save(filename,data):
    with open('JSON/'+filename, 'w') as file_obj:
        json.dump(data, file_obj)

def file_load(filename):
    f = open('JSON/'+filename, encoding='utf-8')
    data = json.loads(f.read())
    return data

def getRankPercent(item1,kg,search_key):
    triple_dict = {}
    v_h = kg[item1]
    for item2 in kg.keys():
        v_t = kg[item2]
        for r in relations:
            v_r = kg[r]
            distance = cal_distance(trans(trans(v_h, v_r, 1), v_t, -1))
            triple_dict[item1 + ' ' + r + ' ' + item2] = distance
    triple_af_sort = sorted(triple_dict.items(), key=lambda item: item[1], reverse=False)
    tmp:list = np.array(triple_af_sort)[:,0].tolist()
    rank = tmp.index(search_key)/len(tmp)
    print(item1, 'over')
    return rank

relations = file_load('relations.json')
kg = file_load('vector_dict_d30.json')
test_list = open('wn18/wordnet-mlj12-test.txt',encoding='utf-8').readlines()

log = []
success = 0
size = 30
correct_rate = 0
it = list(range(0,len(test_list)))
random.shuffle(it)
it = it[:size]
print('起始行',it[0]+1)
for i in it:
    line = test_list[i]
    print('第{}次测试'.format(it.index(i) + 1))
    h,r,t = line.split()
    search_key = h + ' ' + r + ' ' + t
    rank = getRankPercent(h,kg,search_key)
    correct_rate += rank
    if(rank<=0.2):
        success += 1
    s = '{} 所处位置是{:.2f}%'.format(search_key,rank*100)
    print(s)
    log.append(s)
    log.append('\n')

correct_rate = 100-correct_rate/size*100
log.append('滤除率{:.2f}% '.format(correct_rate))
log.append('前20%命中率{:.2f}%'.format(success/size*100))

localtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
f = open('log'+localtime,mode='w',encoding='utf-8')
f.write(''.join(log))


