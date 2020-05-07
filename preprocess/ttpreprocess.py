#!/usr/bin/python
# -*- coding:utf-8 -*-
import re, json, os
import random
import collections
from pyltp import Segmentor
LTP_DIR = "/home/zxsong/workspace/ltp_data_v3.4.0"
segmentor = Segmentor()
segmentor.load(os.path.join(LTP_DIR, "cws.model"))

def split_sentences(article):
    '''
    对文章分句
    :param article: str
    :return: list(str)
    '''
    article = article.strip()
    para = re.sub('([。！!？?\?])([^”’])', r"\1\n\2", article)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！!？?\?][”’])([^，。！!？?\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

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

def preprocess_pipeline(article, cut_level):
    # 分句 分词
    splitted_article = split_sentences(article)  # list(str) 分句
    cutted_article = []
    for sentence in splitted_article:  # list(list(str)) 分词
        # cutted_sentence = self.tokenizer.tokenize(sentence)  # 这里构建UNK, 会将<Paragraph>切分
        cutted_sentence = cut_sentence(sentence.strip(), cut_level=cut_level)
        if len(cutted_sentence) != 0:
            cutted_article.append(cutted_sentence)
    return cutted_article

from tqdm import tqdm

def preprocess_train(cut_level):
    dir_name = "/home/zxsong/workspace/TTNewsCorpus_NLPCC2017/toutiao4nlpcc"
    file_name = os.path.join(dir_name, 'train_with_summ.txt')
    # 分词
    f_x = open("../../../data/toutiao_"+cut_level+"/train/train.article.txt", "w", encoding="utf-8")
    f_y = open("../../../data/toutiao_"+cut_level+"/train/train.title.txt", "w", encoding="utf-8")
    for line in tqdm(open(file_name, 'r', encoding="utf-8")):
        dataset = json.loads(line, encoding="utf-8")
        # key: article, summarization
        article, summarization = dataset['article'], dataset['summarization']  # str, str
        cutted_article = preprocess_pipeline(article, cut_level)
        if len(cutted_article) == 0:
            continue
        cutted_summarization = preprocess_pipeline(summarization, cut_level)
        if len(cutted_summarization) == 0:
            continue
        f_x.write(" ".join([word for cutted_sen in cutted_article for word in cutted_sen]).strip()+"\n")
        f_y.write(" ".join([word for cutted_sen in cutted_summarization for word in cutted_sen]).strip()+"\n")
    f_x.close()
    f_y.close()

def preprocess_dev(cut_level):
    dir_name = "/home/zxsong/workspace/TTNewsCorpus_NLPCC2017/toutiao4nlpcc_eval"
    file_name = os.path.join(dir_name, 'evaluation_with_ground_truth.txt')
    # 分词
    f_x = open("../../../data/toutiao_"+cut_level+"/dev/valid.article.txt", "w", encoding="utf-8")
    f_y = open("../../../data/toutiao_"+cut_level+"/dev/valid.title.txt", "w", encoding="utf-8")
    for line in tqdm(open(file_name, 'r', encoding="utf-8")):
        dataset = json.loads(line, encoding="utf-8")
        # key: article, summarization
        article, summarization = dataset['article'], dataset['summarization']  # str, str
        cutted_article = preprocess_pipeline(article, cut_level)
        if len(cutted_article) == 0:
            continue
        cutted_summarization = preprocess_pipeline(summarization, cut_level)
        if len(cutted_summarization) == 0:
            continue
        f_x.write(" ".join([word for cutted_sen in cutted_article for word in cutted_sen]).strip()+"\n")
        f_y.write(" ".join([word for cutted_sen in cutted_summarization for word in cutted_sen]).strip()+"\n")
    f_x.close()
    f_y.close()

def print_stats(cut_level, split,type):
    lengths = []
    for line in open("../../../data/toutiao_"+cut_level+"/"+split+"/subword/"+split+"."+type+".txt", "r", encoding="utf-8"):
        lengths.append(len(line.strip().split()))
    lengths = list(sorted(lengths, key=lambda x:x, reverse=False))
    print("%50 {}".format(lengths[int(0.5*len(lengths))]))
    print("%75 {}".format(lengths[int(0.75*len(lengths))]))
    print("%90 {}".format(lengths[int(0.9*len(lengths))]))
    print("%95 {}".format(lengths[int(0.95*len(lengths))]))
    print("%96 {}".format(lengths[int(0.96*len(lengths))]))
    print("%97 {}".format(lengths[int(0.97*len(lengths))]))
    print("%98 {}".format(lengths[int(0.98*len(lengths))]))
    print("%99 {}".format(lengths[int(0.99*len(lengths))]))
    print("%100 {}".format(lengths[-1]))
    '''
    word        word+subword+article        word+subword+title
    %50 415         %50 459                     %50 36
    %75 769         %75 845                     %75 40
    %90 1306        %90 1429                    %90 44
    %95 1731        %95 1892                    %95 47
    %96 1867        %96 2060                    %96 47
    %97 2075        %97 2276                    %97 48
    %98 2415        %98 2676                    %98 49
    %99 3144        %99 3450                    %99 51
    
    char 
    %50 687
    %75 1247
    %90 2108
    %95 2789
    %96 3019
    %97 3357
    %98 3921
    %99 5016
    '''

def check_vocab(filename1, filename2):
    def load_vocab(filename):
        vocab = set()
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                vocab.add(sent.split()[0])
        return vocab
    vocab1 = load_vocab(filename1)
    vocab2 = load_vocab(filename2)
    print(len(vocab1))
    print(len(vocab2))
    print(len(vocab2 & vocab1))
    # 110396
    # 39327
    # 34350

def check_sent_num():
    lengths = []
    for line in tqdm(open("/home/zxsong/workspace/seass/data/toutiao_word/train/subword/train.article.txt")):
        line = line.strip()
        sents = split_sentences(line)
        new_sents = []
        for sent in sents:
            if sent[:3] != "@@ ":
                new_sents.append(sent)
            else:
                new_sents[-1] = new_sents[-1] + sent
        sents = new_sents
        lengths.append(len(sents))
    lengths = list(sorted(lengths, key=lambda x: x, reverse=False))
    print("%50 {}".format(lengths[int(0.5 * len(lengths))]))
    print("%75 {}".format(lengths[int(0.75 * len(lengths))]))
    print("%90 {}".format(lengths[int(0.9 * len(lengths))]))
    print("%95 {}".format(lengths[int(0.95 * len(lengths))]))
    print("%96 {}".format(lengths[int(0.96 * len(lengths))]))
    print("%97 {}".format(lengths[int(0.97 * len(lengths))]))
    print("%98 {}".format(lengths[int(0.98 * len(lengths))]))
    print("%99 {}".format(lengths[int(0.99 * len(lengths))]))
    print("%100 {}".format(lengths[-1]))
    # %50 14
    # %75 28
    # %90 47
    # %95 64
    # %96 70
    # %97 78
    # %98 92
    # %99 118
    # %100 541

def preprocess_train_with_semi_supervised(cut_level):
    dir_name = "/home/zxsong/workspace/TTNewsCorpus_NLPCC2017/toutiao4nlpcc"
    # 分词
    f_x = open("../../../data/toutiao_" + cut_level + "/train_with_semi_supervised/train.article.txt", "w", encoding="utf-8")
    f_y = open("../../../data/toutiao_" + cut_level + "/train_with_semi_supervised/train.title.txt", "w", encoding="utf-8")
    for file_name in ['train_with_summ.txt', 'train_without_summ.txt']:
        file_name = os.path.join(dir_name, file_name)
        for line in tqdm(open(file_name, 'r', encoding="utf-8")):
            dataset = json.loads(line, encoding="utf-8")
            # key: article, summarization

            article = dataset['article']  # str
            cutted_article = preprocess_pipeline(article, cut_level)
            if len(cutted_article) == 0:
                continue
            f_x.write(" ".join([word for cutted_sen in cutted_article for word in cutted_sen]).strip()+"\n")

            if "without" not in file_name:
                summarization = dataset['summarization']
                cutted_summarization = preprocess_pipeline(summarization, cut_level)
                if len(cutted_summarization) == 0:
                    continue
                f_y.write(" ".join([word for cutted_sen in cutted_summarization for word in cutted_sen]).strip()+"\n")
    f_x.close()
    f_y.close()

if __name__ == "__main__":
    # gen_sentence_flag()
    # check_vocab("../../../data/toutiao_word/train/subword/source.vocab", "../../../data/toutiao_word/train/subword/target.vocab")
    # preprocess_train(cut_level="word")
    # preprocess_dev(cut_level="word")

    # subword
    print_stats(
        cut_level="word",
        split="train",
        type="article"
    )

    # char target的长度70
    # word target的长度50
    # preprocess_train_with_semi_supervised(cut_level="word")


