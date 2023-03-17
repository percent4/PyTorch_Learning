# -*- coding: utf-8 -*-
# @Time : 2023/3/16 16:42
# @Author : Jclian91
# @File : model_predict.py
# @Place : Minghang, Shanghai
import torch as T
import numpy as np

from text_featuring import load_file_file, text_feature
from model import TextClassifier

model = T.load('sougou_mini_cls.pth')

label_dict, char_dict = load_file_file()
print(label_dict)
text = "曾经在车迷圈子中盛传一句话，“传统品牌要是发力新能源，把特斯拉打得渣儿都不剩”。现在实际情况是，不仅仅是传统汽车品牌的纯电动车被打得节节败退，甚至Model 3(参数|图片)还强力侵蚀着本属于传统豪华品牌的汽油车型市场。"
# text = "北京时间3月16日，NBA官方公布了对于灰熊球星贾-莫兰特直播中持枪事件的调查结果灰熊，由于无法确定枪支是否为莫兰特所有，也无法证明他曾持枪到过NBA场馆，因为对他处以禁赛八场的处罚，且此前已禁赛场次将算在禁赛八场的场次内，他最早将在下周复出。"
# text = "据海上自卫队官方社交媒体消息，当地时间3月7日上午，海上自卫队最上型多用途护卫舰FFM-4“三隈”在三菱重工长崎造船厂交付服役，海上幕僚长（相当于海军参谋长——编辑注）酒井良出席。服役后将与去年底服役的3号舰“能代”一同配属于驻长崎县佐世保基地的第13护卫队。"
# text = "商品属性材质软橡胶带加浮雕工艺+合金彩色队徽吊牌规格162mm数量这一系列产品不限量发行图案软橡胶手机带上有浮雕的队名,配有全彩色合金队徽吊牌用途手机吊饰配件彩色精美纸卡包装.所属球队火箭队所属人物无特殊标志NBA商品介绍将NBA球队的队徽,结合时下最流行的手机吊饰用品,是球迷不可错过的时尚选择.吊饰使用彩色队徽吊牌及软像胶带.并在橡胶带上用球队的主要颜色用浮雕效果做出球队名称,产品都同时搭配彩色NBA标志吊牌,是同时兼具时尚和实用功能的NBA商品商品种类NBA标志手机吊饰及7支球队队徽手机吊饰共8款,(首批推出休士顿火箭队,洛杉矶湖人队,迈阿密热火队,圣安东尼奥马刺队,明尼苏达森林狼队,费城76人队,以及底特律活塞队),其他球队未来将陆续推出."

label, sample = ['汽车'], [text]
samples, y_true = text_feature(label, sample, label_dict, char_dict)
print(samples, y_true)
x = T.from_numpy(np.array(samples)).long()
y_pred = model(x)
print(y_pred)
