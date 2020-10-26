import json
import math
import random
import copy
import sys
import time

def file_save(filename,data):
    with open('JSON/'+filename, 'w') as file_obj:
        json.dump(data, file_obj)

def file_load(filename):
    f = open('JSON/'+filename, encoding='utf-8')
    data = json.loads(f.read())
    return data

def trans(vec1:list, vec2:list, rate):
    if(len(vec1)!=len(vec2)):
        return Exception
    res = []
    for l in range(len(vec1)):
        res.append(vec1[l] + vec2[l]*rate)
    return res

def cal_distance(vec:list):
    # 取平方
    res = 0
    for i in vec:
        res += i**2
    return res

# def grad(vec:list,learn_rate):
#     # 梯度下降
#     # 操作对象是一个向量
#     res = []
#     for i in range(len(vec)):
#         res.append((1+(-2) * learn_rate ) * vec[i])
#     return res

def norm(vec):
    k = math.sqrt(1 / cal_distance(vec))
    return trans(vec, vec, k - 1)


def hrt(h, r, t):
    return cal_distance(trans(trans(h, r, 1), t, -1))


class KG_TansE:
    def __init__(self,heads,relations,tails,vector_d):
        self.heads = heads
        self.relations = relations
        self.tails = tails
        self.size = len(heads)
        # 储存三元组，注意放的是key（在items_dict中）
        self.items_set = list(set(self.heads + self.relations + self.tails))
        self.items_dict = dict(zip(self.items_set,[1]*len(self.items_set)))
        for item in self.items_dict.keys():
            self.items_dict[item] = []
            for i in range(vector_d):
                self.items_dict[item].append(random.random())
            self.items_dict[item] = norm(self.items_dict[item])
        # 实体和关系的字典（向量字典）
        # 随机初始向量
        self.wrong_heads = copy.deepcopy(self.heads)
        self.wrong_relations = copy.deepcopy(self.relations)
        self.wrong_tails = copy.deepcopy(self.tails)
        # 初始化错误三元组
        self.learn_rate = 0
        self.margin = 0
        self.loss_now = 0
        self.train_times_max = 0
        self.depth = 0
        self.depth_now = 0
        self.vector_d =vector_d

    def pram_set(self,learn_rate,margin,train_times_max,depth):
        # learn_rate 学习率
        # margin 允许的词向量误差
        self.learn_rate = learn_rate
        self.margin = margin
        self.loss_now = sys.float_info.max
        self.train_times_max = train_times_max
        self.__make_wrong_triple()
        self.depth = depth

    def __make_wrong_triple(self):
        t1 = copy.deepcopy(self.heads)
        t2 = copy.deepcopy(self.tails)
        random.shuffle(t1)
        random.shuffle(t2)
        self.wrong_relations = self.relations
        for i in range(self.size):
            if(random.randint(1,2)==1):
                self.wrong_heads[i] = t1[i]
            else:
                self.wrong_tails[i] = t2[i]
            # 换头或换尾
        # 生成错误三元组
        # 出于效率考虑，没有进行存在检测

    def __modify(self):
        # 尝试更新一次数据
        # self.__make_wrong_triple()
        loss = 0
        for i in range(self.size):
            hi = self.heads[i]
            ri = self.relations[i]
            ti = self.tails[i]
            h = self.items_dict[hi]
            r = self.items_dict[ri]
            t = self.items_dict[ti]
            loss_single = hrt(h,r,t)
            h_ = self.items_dict[self.wrong_heads[i]]
            r_ = self.items_dict[self.wrong_relations[i]]
            t_ = self.items_dict[self.wrong_tails[i]]
            win_single = hrt(h_,r_,t_)

            if (self.margin + loss_single - win_single >= 0):
                if (hi == self.wrong_heads[i]):
                    # 头部一样
                    # Δh = (-2t + 2t') * learn_rate
                    # Δr = (-2t + 2t') * learn_rate
                    # Δt = (-2h -2r + 2t) * learn_rate
                    h_det = trans(t_, t, -1)
                    h_det = trans(h_det, h_det, 2 * self.learn_rate - 1)
                    r_det = trans(t_, t, -1)
                    r_det = trans(h_det, h_det, 2 * self.learn_rate - 1)
                    t_det = trans(trans(t, r, -1), h, -1)
                    t_det = trans(t_det, t_det, 2 * self.learn_rate - 1)
                    h = trans(h, h_det, -1)
                    r = trans(r, h_det, -1)
                    t = trans(t, h_det, -1)
                    h_ = h
                    r_ = r
                else:
                    # 尾部一样
                    # Δh = (2h + 2r - 2t) * learn_rate
                    # Δr = (2h - 2h') * learn_rate
                    # Δt = (-2h + 2h') * learn_rate
                    h_det = trans(trans(h, r, 1), t, -1)
                    h_det = trans(h_det, h_det, 2 * self.learn_rate - 1)
                    r_det = trans(h, h_, -1)
                    r_det = trans(r_det, r_det, 2 * self.learn_rate - 1)
                    t_det = trans(h_, h, -1)
                    t_det = trans(t_det, t_det, 2 * self.learn_rate - 1)
                    h = trans(h, h_det, -1)
                    r = trans(r, r_det, -1)
                    t = trans(t, t_det, -1)
                    r_ = r
                    t_ = t

                self.items_dict[hi] = norm(h)
                self.items_dict[ri] = norm(r)
                self.items_dict[ti] = norm(t)

                loss_single_af = hrt(h, r, t)
                win_single_af = hrt(h_,r_,t_)
                loss_single  = loss_single_af
                win_single = win_single_af

            loss += loss_single-win_single

        print('当前的损失是', loss)

        if(self.loss_now==loss or self.depth_now==self.depth):
            self.__make_wrong_triple()
            print('负例三元组已更换')
            self.depth_now = 0
            print('深度重设')

        self.loss_now = loss
        self.depth_now += 1


    def train(self):
        for i in range(self.train_times_max):
            self.__modify()

            print('当前训练次数',i)
            

if __name__ == '__main__':
    random.seed(0)
    localtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print('开始时间',localtime)

    f = open('wn18/wordnet-mlj12-train.txt',encoding='utf-8')
    lines = f.readlines()
    h = []
    r = []
    t = []
    for line in lines:
        h_,r_,t_ = line.split()
        h.append(h_)
        r.append(r_)
        t.append(t_)

    kg = KG_TansE(h,r,t,20)

    kg.pram_set(learn_rate=0.1,margin=0.6,train_times_max=40,depth=10)
    kg.train()
    file_save('vector_dict_fs.json',kg.items_dict)
    print('ok')