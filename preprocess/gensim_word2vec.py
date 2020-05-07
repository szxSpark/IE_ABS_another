from gensim.models import Word2Vec
import os
import re, multiprocessing
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def split_sentences(article):
    '''
    对文章分句
    :param article: str
    :return: list(str)
    '''
    article = article.strip()
    article = article.replace("。@@", "@@@@@@@@")
    para = re.sub('([。！!？?\?])([^”’])', r"\1\n\2", article)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！!？?\?][”’])([^，。！!？?\?])', r'\1\n\2', para)
    para = para.rstrip()
    para = para.replace("@@@@@@@@", "。@@")
    return para.split("\n")

def train(file_name, save_file, min_count):
    sentences = []
    for line in tqdm(open(file_name)):
        for sent in split_sentences(line.strip()):
            if len(sent.split(' ')) > 2:
                sent = [w for w in sent.split(' ') if len(w) > 0]
                if len(sent) > 0:
                    sentences.append(sent)
    print(len(sentences))
    # 负采样，skip-gram
    model = Word2Vec(sentences, sg=1, hs=0, min_count=min_count, iter=30, size=512, workers=multiprocessing.cpu_count())
    print(len(model.wv.vocab.keys()))
    model.save(save_file)

    a = set(model.wv.vocab.keys())
    if "title" in file_name:
        b = set([line.split(" ")[0] for line in
             open("/home/zxsong/workspace/seass/data/toutiao_word/train/subword/target.vocab")])
    else:
        b = set([line.split(" ")[0] for line in
                 open("/home/zxsong/workspace/seass/data/toutiao_word/train/subword/source.vocab")])
    print(len(a), len(b), len(a & b))
    print(a - b)
    print(b - a)
    # print(word, type(model.wv[word]))

import numpy as np

def save_npy(vocab_file, save_file):
    vocab = [line.split(" ")[0] for line in open(vocab_file)]
    model = Word2Vec.load(save_file)
    print(len(model.wv.vocab.keys()))
    embedding = []
    for word in vocab:
        if word in model.wv:
            print(word)
            embedding.append(model.wv[word].tolist())
        else:
            print("random", word)
            embedding.append(np.random.normal(size=512).tolist())
    embedding = np.array(embedding)
    print(np.shape(embedding))  # (28755, 512)
    if "source" in vocab_file:
        np.save(os.path.join(os.path.dirname(vocab_file), "source_pretrained_vectors.npy"), embedding)
    else:
        np.save(os.path.join(os.path.dirname(vocab_file), "target_pretrained_vectors.npy"), embedding)

if __name__ == "__main__":
    # file_name = "/home/zxsong/workspace/seass/data/toutiao_word/train_with_semi_supervised/subword/train.article.BPE.txt"
    save_file = "./word2vec/target.word2vec.model"
    train(
        # file_name="/home/zxsong/workspace/seass/data/toutiao_word/train/subword/train.title.txt",
        file_name="/home/zxsong/workspace/seass/data/lcsts/train/subword/train.title.txt",
        save_file=save_file,
        min_count=2 # 5, 2
    )
    save_npy(
        vocab_file="/home/zxsong/workspace/seass/data/lcsts/train/subword/target.vocab",
        save_file=save_file
    )
