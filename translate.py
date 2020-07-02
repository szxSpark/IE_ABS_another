from __future__ import division
import argparse
import torch
import torch.nn as nn
from models import Optim, NMTModel, Encoder, Decoder, DecInit, Generator
from torch import cuda
import math
import time
import logging
import numpy as np
import random
from PyRouge.Rouge import Rouge
import xargs
from utils import onlinePreprocess
from utils.onlinePreprocess import prepare_data_online
import utils.Constants as Constants
from utils.Translator import Translator
from utils.Dataset import Dataset
import os
import re
from utils.Constants import E1_R_WORD, R_E2_WORD
import sys
from tqdm import tqdm
import json
from oie_extraction import extract_elements

from pyltp import Segmentor
LTP_DIR = "/home/user-2/zxsong_github/MonitorSystem/lib/ltp_data_v3.4.0"
segmentor = Segmentor()
segmentor.load(os.path.join(LTP_DIR, "cws.model"))

def cut_sentence(sentence, cut_level="char"):
    '''
    对句子分词，采用字级别的分词方式
    :param sentence: str
    :return: list(str)
    '''
    # TTnews含有特殊符号<Paragraph>，将该符号作为特殊token
    cutted = []
    for _sentence in sentence.split("<Paragraph>"):
        _sentence = _sentence.strip()
        if cut_level == "char":
            for _token in _sentence:
                if len(_token.strip()) == 0:
                    continue
                cutted.append(_token.strip())
        else:
            _sentence = list(segmentor.segment(_sentence))
            _sentence = [w.strip() for w in _sentence if len(w.strip())>0]
            cutted.extend(_sentence)
        cutted.append("<Paragraph>")
    if len(cutted) !=0 and cutted[-1] == "<Paragraph>":
        del cutted[-1]
    return cutted

def split_sentences(article):
    '''
    对文章分句
    :param article: str
    :return: list(str)
    '''
    article = article.strip()
    sentences = re.split('(。|！|!|\!|\?|？|\?)', article)  # 保留分割符
    if len(sentences) == 1:
        return sentences
    new_sents = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        new_sents.append(sent)
    return new_sents

def preprocess_pipeline(article, cut_level):
    # 分句 分词
    splitted_article = split_sentences(article)  # list(str) 分句
    cutted_article = []
    for sentence in splitted_article:  # list(list(str)) 分词
        cutted_sentence = cut_sentence(sentence.strip(), cut_level=cut_level)
        if len(cutted_sentence) != 0:
            cutted_article.append(cutted_sentence)
    return cutted_article

def shell_subword(data, in_f, out_f):
    # data:只是一条数据 str
    assert type(data) == list
    with open(in_f, "w", encoding="utf-8")as f:
        for one_data in data:
            f.write(str(one_data.strip())+"\n")
    codes_file = "./subword/codes"
    voc_file = "./subword/voc.article"
    cmd = "subword-nmt apply-bpe -c {} --vocabulary {} --vocabulary-threshold 50 < {} > {}".format(
        codes_file, voc_file, in_f, out_f
    )
    os.system(cmd)
    with open(out_f, "r", encoding="utf-8")as f:
        return f.readlines()

def magic_data_process(translator, article):
    # 先分句，再分词
    cut_level = "word"
    cutted_article = preprocess_pipeline(article, cut_level)
    cutted_article_str = " ".join([word for cutted_sen in cutted_article for word in cutted_sen]).strip()
    # 这里要subword
    subword_article = shell_subword([cutted_article_str], in_f="./subword/inf.tmp.txt", out_f="./subword/outf.tmp.txt")
    src_tokens = "".join([s.strip() for s in subword_article]).strip().split(' ')[:2000]

    # 采用融合要素抽取的模型，需要计算oie
    spo_list = extract_elements(article, LTP_DIR)

    # 构建subword的spo
    spo_words = []
    for e1, r, e2 in spo_list:
        e1 = [" ".join(e1)]
        r = [" ".join(r)]
        e2 = [" ".join(e2)]
        spo_words.append(e1 + r + e2)
    spo_words = ["\n".join(spo) for spo in spo_words]
    subword_spo = shell_subword(spo_words, in_f="./subword/inf.spo.tmp.txt", out_f="./subword/outf.spo.tmp.txt")

    # 逆操作
    c = 0
    one_spo = ""
    final_spo = []
    for line in subword_spo:
        c += 1
        one_spo += line.strip()
        if c % 3 == 0:
            # 一个spo
            final_spo.append(one_spo)
            one_spo = ""
        elif c % 3 == 1:
            # s
            one_spo += " " + E1_R_WORD + " "
        else:
            # s, v
            one_spo += " " + R_E2_WORD + " "
    assert c % 3 == 0

    # 构建数据, batch_size=1
    src_batch = [src_tokens]
    spo_list = [one_spo.split(" ") for one_spo in final_spo]
    spo_batch = [spo_list]

    data = translator.buildData(src_batch, None, spo_batch)
    return data

def compute_loss(g_outputs, g_targets, generator, crit, g_p_gens, g_attn, enc_batch_extend_vocab, extra_zeros, coverage_losses):
    # g_outputs: (decL-1, B, H)
    # g_targets: (decL-1, B)
    # g_p_gens:  (decL-1, B, 1)
    # coverage_losses  (decL-1, B)
    if opt.pointer_gen:
        dec_padding_mask = 1 - g_targets.eq(Constants.PAD)  # (decL-1, B)
        dec_padding_mask = dec_padding_mask.view(-1).float()  # (decL-1*B,)
        g_prob_t = generator(g_outputs, g_p_gens, g_attn, enc_batch_extend_vocab, extra_zeros)  # (decL-1*B, tgt_size)
        target = g_targets.view(-1)  # (decL-1*B,)
        gold_probs = torch.gather(g_prob_t, 1, target.unsqueeze(1)).squeeze()  # (decL-1*B,)  目的是拿到金的
        step_loss = -torch.log(gold_probs + 1e-12)  # (decL-1*B,)  generator应该是sm
        nll_loss = torch.sum(step_loss * dec_padding_mask).item()
        if opt.is_coverage:
            step_loss = step_loss + coverage_losses.view(-1)  # (decL-1*B,)
        step_loss = step_loss * dec_padding_mask  # (decL-1*B,)
        total_loss = torch.sum(step_loss)
        report_loss = total_loss.item()
        if opt.is_coverage:
            coverage_report_loss = torch.sum(coverage_losses.view(-1)*dec_padding_mask).item()
            return total_loss, (report_loss, nll_loss, coverage_report_loss), 0
        return total_loss, report_loss, 0
    else:
        g_out_t = g_outputs.view(-1, g_outputs.size(2))
        g_prob_t = generator(g_out_t)
        g_loss = crit(g_prob_t, g_targets.view(-1))
        total_loss = g_loss
        report_loss = total_loss.item()
        return total_loss, report_loss, 0

def load_model(model_file):
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    vocab_dicts = checkpoint['dicts']
    opt = checkpoint['opt']
    opt.gpus = []  # 清除gpu设置
    encoder = Encoder(opt, vocab_dicts['src'])
    decoder = Decoder(opt, vocab_dicts['tgt'])
    if opt.share_embedding:
        decoder.word_lut = encoder.word_lut
    decIniter = DecInit(opt)
    model = NMTModel(encoder, decoder, decIniter)
    model.load_state_dict(checkpoint['model'])
    if opt.pointer_gen:
        generator = Generator(opt, vocab_dicts)
    else:
        generator = nn.Sequential(
            nn.Linear(opt.dec_rnn_size // opt.maxout_pool_size, vocab_dicts['tgt'].size()),
            # nn.Linear(opt.word_vec_size, dicts['tgt'].size()),  # transformer
            nn.LogSoftmax(dim=-1)
        )
    generator.load_state_dict(checkpoint['generator'])
    model.generator = generator
    return model, opt, vocab_dicts

def predict_oneData(translator, data):
    batch = data[0]
    # for batch in data:
    src, _, indices = batch

    #  (2) translate
    with torch.no_grad():
        pred, predScore, attn, _ = translator.translateBatch(src)  # 无teacher forcing
        pred, predScore, attn = list(zip(
            *sorted(zip(pred, predScore, attn, indices),
                    key=lambda x: x[-1])))[:-1]
    #  (3) convert indexes to words
    predBatch = []
    for b in range(src[0].size(1)):  # B
        n = 0
        predBatch.append(
            translator.buildTargetTokens(pred[b][n], None, attn[b][n])
        )
    predict = [' '.join(sents) for sents in predBatch][0]  # str

    tmp_file = "./subword/predict.txt"
    with open(tmp_file, 'w', encoding='utf-8') as of:
        of.write(predict.strip() + '\n')

    # cmd = "sed -r 's/(@@ )|(@@ ?$)//g' {} > {}".format(tmp_file, ofn)  # 写的是分词的
    cmd = "sed -r 's/(@@ )|(@@ ?$)//g' {}".format(tmp_file)  # 写的是分词的
    fouput = os.popen(cmd)
    result = fouput.readlines()
    print("result is: %s" % result)
    # os.remove(ofn + ".tmp")
    # predict = []
    # # opt.dev_ref 不要用subword
    # with open(ofn, encoding='utf-8') as f:
    #     for line in f:
    #         if not line:
    #             break
    #         # predict.append(line.strip())
    #         tmp_line = " ".join(list("".join(line.strip().split(" "))))
    #         tmp_line = tmp_line.replace("< u n k >", "<unk>")
    #         predict.append(tmp_line)  # 按照char计算指标


def main():
    model_file = "./checkpoints/model_devRouge_0.6756_0.2952_e62.pt"
    model, opt, vocab_dicts = load_model(model_file)
    translator = Translator(opt, model, vocab_dicts)

    # article = '温州网讯昨天，广受关注的“瑞安孕妇重度烧伤”一事有了新进展：家属称，王芙蓉现阶段治疗急需A型血小板；事故初步认定为由燃气泄漏引起的爆炸事故。5月9日晚11时许，怀孕8个月的王芙蓉和母亲在厨房煮夜宵时，厨房里突然发生爆燃，一家四口不同程度烧伤。据悉，目前捐给王芙蓉一家的爱心款累计突破600万元。王芙蓉已有较清晰的意识王芙蓉的叔叔王先生说，5月20日，王芙蓉做了清理坏死皮肤的手\n术，术后恢复情况较为理想，虽还不能说话，但已有较清晰的意识。王芙蓉的丈夫和她的父母也恢复得不错。但是让家属担心的是，前天医院\n通知说王芙蓉需要A型血小板，但目前瑞安市血库存量不多，这几天他们正发动亲戚朋友捐A型血小板。瑞安市血站得知这一情况后，第一时间\n通知志愿者前来献血。据血站的相关人员介绍，目前已有4名志愿者提供血小板，可满足本周用量，但如果下周一还是需要血小板的话，就需要社会上的好心人继续献出爱心。据介绍，想为王芙蓉捐A型血小板的热心市民，可到瑞安市血站献血，也可直接到市中心血站献血，献血时向工作人员表明是献给王芙蓉即可。调查组初步认定4点结果事发后，瑞安立刻成立事故调查组展开调查。据事故调查组组长、瑞安市安监局副局长顾荣华介绍，目前该起事故已有初步认定结果：一是此次事故的性质已基本确认为燃气泄漏引起的爆炸事故；二是爆炸部位也已基本确认，是\n位于厨房间洗碗水槽下方的一个密闭的空间；三是爆炸主因，燃气泄漏扩散蔓延，与空气混合成爆炸性气体，当浓度达到爆炸极限下限时，遇\n到点火源，瞬间产生爆炸；四是排除由户外管道燃气泄漏引起爆炸的可能性。据调查组介绍，下一步将根据相关规定查清事故责任，主要是要\n查明这起事故是不是生产安全责任事故。本文转自：温州网'  # str
    article = '四海网讯，\n近日，有媒体报道称：章子怡真怀孕了!报道还援引知情人士消息称，“章子怡怀孕大概四五个月，预产期是年底前后，现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息，23日晚8时30分，华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士，这位人士向华西都市报记者证实说：“子怡这次确实怀孕了。她已经36岁了，也该怀孕了。章子怡怀上汪峰的孩子后，子怡的父母亲十分高兴。子怡的母亲，已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时，华西都市报记者为了求证章子怡怀孕消息，又电话联系章子怡的亲哥\n哥章子男，但电话通了，一直没有人<Paragraph>接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来，就被传N遍了!不过，\n时间跨入2015年，事情却发生着微妙的变化。2015年3月21日，章子怡担任制片人的电影《从天儿降》开机，在开机发布会上几张合影，让网友又燃起了好奇心：“章子怡真的怀孕了吗?”但后据证实，章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日，《太平轮》新一轮宣\n传，章子怡又被发现状态不佳，不时深呼吸，不自觉想捂住肚子，又觉得不妥。然后在8月的一天，章子怡和朋友吃饭，在酒店门口被风行工作室拍到了，疑似有孕在身!今年7月11日，汪峰本来在上海要举行演唱会，后来因为台风“灿鸿”取消了。而消息人士称，汪峰原来打算在演唱会\n上当着章子怡的面宣布重大消息，而且章子怡已经赴上海准备参加演唱会了，怎知遇到台风，只好延期，相信9月26日的演唱会应该还会有惊喜大白天下吧。'
    data = magic_data_process(translator, article)  # 已经分好batch Dataset， list((src_batch, tgt_batch)))
    predict_oneData(translator, data)

if __name__ == "__main__":
    main()
