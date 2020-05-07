import logging
import re
from PyRouge.Rouge import Rouge
from PerlRouge.utils import compute_rouge

report_every = 1000

logger = logging.getLogger(__name__)

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

def makeData(srcFile, tgtFile):
    count, ignored = 0, 0
    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))

    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')
    src_sents, tgt_sents = [], []
    while True:
        sline = srcF.readline().strip()
        tline = tgtF.readline().strip()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            logger.info('WARNING: source and target do not have the same number of sentences')
            break

        # source and/or target are empty
        if sline == "" or tline == "":
            logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue


        src_sents.append(split_sentences(sline))
        tgt_sents.append(tline)

        count += 1
        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()
    return src_sents, tgt_sents

def lead(src_sents, is_cn):
    # 第一个句子
    pred = list(map(lambda x:x[0], src_sents))
    with open("./myPyrouge/lead_lcsts/lead.0.txt", "w", encoding="utf-8")as f:
        for line in pred:
            line = line.replace("<Paragraph>", " ")
            if is_cn:
                line = " ".join(list("".join(line.split(" "))))
            f.write(line.strip()+"\n")

rouge_calculator = Rouge.Rouge()
from tqdm import tqdm
def oracle(src_sents, tgt_sents, sent_n=1, is_cn=True):
    class Score(object):
        def __init__(self, scores):
            self.rouge_1 = scores['rouge-1']
            self.rouge_2 = scores['rouge-2']
            self.rouge_l = scores['rouge-l']

        def __lt__(self, other):
            # return self.rouge_l < other.rouge_l
            # return self.rouge_1 < other.rouge_1
            # return self.rouge_2 < other.rouge_2
            return self.rouge_1 + self.rouge_2 + self.rouge_l < other.rouge_1 + other.rouge_2 + other.rouge_l

        def __repr__(self):
            return "rouge_1: {}, rouge_2:{}, rouge_l:{}".format(self.rouge_1, self.rouge_2, self.rouge_l)

    final_pred = [[] for _ in src_sents]
    final_score = [None]*len(src_sents)
    for i, (src_multi_sent, gold) in tqdm(enumerate(zip(src_sents, tgt_sents))):
        # 遍历每条数据
        gold = gold.strip()  # str
        while len(final_pred[i]) < min(sent_n, len(src_multi_sent)):  # 继续加入新的
            max_score = final_score[i]
            new_sent = None
            new_sent_idx = None
            for j, sent in enumerate(src_multi_sent):
                # 尝试加入新的sent，只选择最高提升的
                sent = sent.replace("<Paragraph>", " ")
                sent = " ".join(list("".join(sent.split(" "))))
                pred = " ".join(final_pred[i] + [sent])
                # 判断加入后，是否得分提升
                # scores = rouge_calculator.compute_rouge([gold], [pred])  # rouge-1 rouge-2
                scores = compute_rouge(
                    summaries=[
                        [pred]
                    ],
                    references=[
                        [[gold]]
                    ]
                )
                now_score = Score(scores)
                if max_score == None or now_score > max_score:
                    max_score = now_score
                    new_sent, new_sent_idx = sent, j
            if new_sent_idx == None:
                # 加任何的都不work
                break
            else:
                del src_multi_sent[new_sent_idx]
                final_pred[i] += [new_sent]
                final_score[i] = max_score
    # final_pred = [" ".join(a) for a in final_pred]
    # scores = rouge_calculator.compute_rouge(tgt_sents, final_pred)
    # print(scores)

    # 0: rouge-l
    # 1: rouge-1
    # 2: rouge-2
    # 3: avg  # 最优

    with open("./myPyrouge/oracle_lcsts/oracle.3.txt", "w", encoding="utf-8")as f:
        for pred in final_pred:
            pred = " ".join(pred)
            f.write(pred.strip() + "\n")

def compute_length(filename):
    count = 0
    total = 0
    for line in open(filename):
        count += 1
        total += len(line.strip().split(" "))
    print(float(total)/count)

def statics_length():
    compute_length("/home/zxsong/workspace/seass/data/toutiao_word/train/subword/train.article.txt")
    compute_length("/home/zxsong/workspace/seass/data/toutiao_word/train/subword/train.title.txt")
    compute_length("/home/zxsong/workspace/seass/data/lcsts/train/subword/train.article.txt")
    compute_length("/home/zxsong/workspace/seass/data/lcsts/train/subword/train.title.txt")
    compute_length("/home/zxsong/workspace/seass/data/giga/train/train.article.txt")
    compute_length("/home/zxsong/workspace/seass/data/giga/train/train.title.txt")


if __name__ == "__main__":
    # src_sents, tgt_sents = makeData(
    #     srcFile="/home/zxsong/workspace/seass/data/lcsts/dev/valid.article.txt",
    #     tgtFile="/home/zxsong/workspace/seass/data/lcsts/dev/valid.title.char.txt"
    # )
    # is_cn = True
    # # lead(src_sents, is_cn)
    # oracle(src_sents, tgt_sents, sent_n=1, is_cn=is_cn)

    statics_length()