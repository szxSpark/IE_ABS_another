import random
import os, re
from tqdm import tqdm
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

def cut_sentence(sentence, cut_level):
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


def extract_LCSTS(origin, is_partI=False):
    if is_partI:
        tmp = 0
    else:
        tmp = 1

    data = []

    with open(origin) as f_origin:
        lines = f_origin.read().splitlines()
        for i in range(0, len(lines), 8+tmp):
            if not is_partI:
                score_line = lines[i+1].strip()
                if int(score_line[13]) < 3:
                    continue
            data.append((lines[i+5+tmp].strip(), lines[i+2+tmp].strip()))
    return data

def save_data(x, y, output_dir, prefix):
    with open("{}/{}.target".format(output_dir, prefix), 'w') as tgt_output, open("{}/{}.source".format(output_dir, prefix), 'w') as src_output:
        tgt_output.write('\n'.join(y))
        src_output.write('\n'.join(x))

def process_original_data():
    # Arguments
    PART_I_data = '/home/zxsong/workspace/LCSTS_ORIGIN/DATA/PART_I.txt'
    PART_II_data = '/home/zxsong/workspace/LCSTS_ORIGIN/DATA/PART_II.txt'
    PART_III_data = '/home/zxsong/workspace/LCSTS_ORIGIN/DATA/PART_III.txt'
    output_dir = '/home/zxsong/workspace/seass/data/lcsts/clean_data/'

    # Extract data
    partI_data = extract_LCSTS(PART_I_data, is_partI=True)
    partII_data = extract_LCSTS(PART_II_data)
    partIII_data = extract_LCSTS(PART_III_data)

    # print(len(partI_data & partII_data))
    # print(len(partI_data & partIII_data))
    # print(len(partII_data & partIII_data))
    # print(len(partI_data & partII_data & partIII_data))

    # article 作为key    summary作为key     both key
    # 6401                  4772            3892
    # 185                   158             0
    # 54                    58              54
    # 18                    19              0

    # Remove overlapping data
    overlap_cnt = 0
    clean_partI_data = []
    for idx in range(len(partI_data)):
        if partI_data[idx] in partIII_data:
            overlap_cnt += 1
        else:
            clean_partI_data.append(partI_data[idx])

    dirname = os.path.dirname(output_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    clean_partI_x, clean_partI_y = zip(*clean_partI_data)
    save_data(clean_partI_x, clean_partI_y, output_dir, 'train')
    partIII_x, partIII_y = zip(*partIII_data)
    save_data(partIII_x, partIII_y, output_dir, 'test')
    print(len(partII_data))

    print("Remove {} pairs".format(overlap_cnt))

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

def preprocess_train():
    dir_name = "/home/zxsong/workspace/seass/data/lcsts/clean_data"
    # 分词
    f_x = open("../../../data/lcsts/train/train.article.txt", "w", encoding="utf-8")
    f_y = open("../../../data/lcsts/train/train.title.txt", "w", encoding="utf-8")
    for article, summarization in tqdm(zip(open(os.path.join(dir_name, 'train.source'), 'r', encoding="utf-8"),
                    open(os.path.join(dir_name, 'train.target'), 'r', encoding="utf-8"))):
        article = article.strip()
        summarization = summarization.strip()
        cutted_article = preprocess_pipeline(article, cut_level="word")
        if len(cutted_article) == 0:
            continue
        cutted_summarization = preprocess_pipeline(summarization, cut_level='word')
        if len(cutted_summarization) == 0:
            continue
        f_x.write(" ".join([word for cutted_sen in cutted_article for word in cutted_sen]).strip()+"\n")
        f_y.write(" ".join([word for cutted_sen in cutted_summarization for word in cutted_sen]).strip()+"\n")
    f_x.close()
    f_y.close()

def preprocess_dev():
    dir_name = "/home/zxsong/workspace/seass/data/lcsts/clean_data"
    # 分词
    f_x = open("../../../data/lcsts/dev/valid.article.txt", "w", encoding="utf-8")
    f_y = open("../../../data/lcsts/dev/valid.title.txt", "w", encoding="utf-8")
    for article, summarization in tqdm(zip(open(os.path.join(dir_name, 'test.source'), 'r', encoding="utf-8"),
                                      open(os.path.join(dir_name, 'test.target'), 'r', encoding="utf-8"))):
        article = article.strip()
        summarization = summarization.strip()
        cutted_article = preprocess_pipeline(article, cut_level="word")
        if len(cutted_article) == 0:
            continue
        cutted_summarization = preprocess_pipeline(summarization, cut_level="word")
        if len(cutted_summarization) == 0:
            continue
        f_x.write(" ".join([word for cutted_sen in cutted_article for word in cutted_sen]).strip()+"\n")
        f_y.write(" ".join([word for cutted_sen in cutted_summarization for word in cutted_sen]).strip()+"\n")
    f_x.close()
    f_y.close()

def preprocess_dev_char():
    dir_name = "/home/zxsong/workspace/seass/data/lcsts/clean_data"
    # 分词
    f_y = open("../../../data/lcsts/dev/valid.title.char.txt", "w", encoding="utf-8")
    for summarization in tqdm(open(os.path.join(dir_name, 'test.target'), 'r', encoding="utf-8")):
        summarization = summarization.strip()
        cutted_summarization = preprocess_pipeline(summarization, cut_level="char")
        if len(cutted_summarization) == 0:
            continue
        f_y.write(" ".join([word for cutted_sen in cutted_summarization for word in cutted_sen]).strip() + "\n")
    f_y.close()

def print_stats(type):
    lengths = []
    for line in open("../../../data/lcsts/train/subword/train." + type + ".txt", "r",
                     encoding="utf-8"):
        lengths.append(len(line.strip().split()))
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
    '''
    word+subword+article        word+subword+title
        %50 69                      %50 12
        %75 75                      %75 14
        %90 81                      %90 16
        %95 84                      %95 17
        %96 85                      %96 18
        %97 87                      %97 18
        %98 88                      %98 19
        %99 91                      %99 20
        %100 130                    %100 29
    '''

if __name__ == '__main__':
    # preprocess_train()
    # preprocess_dev()
    # preprocess_dev_char()
    print_stats(
        type="article"
    )
    print_stats(
        type="title"
    )