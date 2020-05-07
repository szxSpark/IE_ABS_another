import logging
import torch
import re
import utils.Constants as Constants
from utils.Dict import Dict
import json

lower = True
seq_length = None  # english
report_every = 1000
shuffle = 1
logger = logging.getLogger(__name__)

def makeVocabulary(filenames, size):
    vocab = Dict([Constants.PAD_WORD, Constants.UNK_WORD,
                         Constants.BOS_WORD, Constants.EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    logger.info('Created dictionary of size %d (pruned from %d)' %
                (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = Dict()
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.lookup(Constants.UNK_WORD)
    for w in article_words:
        i = vocab.lookup(w, unk_id)  # 查不到默认unk
        if i == unk_id:  # oov
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.lookup(Constants.UNK_WORD)
    for w in abstract_words:
        i = vocab.lookup(w, unk_id)  # 查不到默认unk
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids

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

def makeData(srcFile, tgtFile, srcDicts, tgtDicts, svoFile, pointer_gen=False):
    src, tgt = [], []
    svo = []
    sizes = []
    src_extend_vocab, tgt_extend_vocab = [], []
    src_oovs_list = []
    count, ignored = 0, 0
    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))

    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')
    svoF = open(svoFile, encoding='utf-8')

    while True:
        sline = srcF.readline().strip()
        tline = tgtF.readline().strip()
        svoline = svoF.readline().strip()

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

        srcWords = sline.split(' ')
        tgtWords = tline.split(' ')
        srcWords = srcWords[:seq_length]
        svo_list = [one_svo.split(" ") for one_svo in json.loads(svoline, encoding="utf-8")]

        src += [srcDicts.convertToIdx(srcWords,
                                      Constants.UNK_WORD)]  # [Tensor]
        tgt += [tgtDicts.convertToIdx(tgtWords,
                                      Constants.UNK_WORD,
                                      Constants.BOS_WORD,
                                      Constants.EOS_WORD)]  # 添加特殊token
        svo += [[srcDicts.convertToIdx(one_svo, Constants.UNK_WORD) for one_svo in svo_list]]
        sizes += [len(srcWords)]
        if pointer_gen:
            # 存储临时的oov词典
            enc_input_extend_vocab, article_oovs = article2ids(srcWords, srcDicts)
            abs_ids_extend_vocab = abstract2ids(tgtWords, tgtDicts, article_oovs)
            # 覆盖target，用于使用临时词典
            vec = []
            vec += [srcDicts.lookup(Constants.BOS_WORD)]
            vec += abs_ids_extend_vocab
            vec += [srcDicts.lookup(Constants.EOS_WORD)]
            src_extend_vocab += [enc_input_extend_vocab]
            src_oovs_list += [article_oovs]
            tgt_extend_vocab.append(torch.LongTensor(vec))

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    logger.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    svo = [svo[idx] for idx in perm]

    if pointer_gen:
        src_extend_vocab = [src_extend_vocab[idx] for idx in perm]
        tgt_extend_vocab = [tgt_extend_vocab[idx] for idx in perm]
        src_oovs_list = [src_oovs_list[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, seq_length))

    return (src, svo), tgt, (src_extend_vocab, tgt_extend_vocab, src_oovs_list)  # list(Tensor)


def prepare_data_online(opt):
    train_src, src_vocab, train_tgt, tgt_vocab, pointer_gen = opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab, opt.pointer_gen
    svoFile = opt.train_svo
    vocab_dicts = {}
    vocab_dicts['src'] = initVocabulary('source', [train_src], src_vocab, 0)
    vocab_dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, 0)

    logger.info('Preparing training ...')
    train_data = {}
    train_data['src'], train_data['tgt'], (train_data['src_extend_vocab'], train_data['tgt_extend_vocab'], train_data['src_oovs_list'])\
        = makeData(train_src, train_tgt, vocab_dicts['src'], vocab_dicts['tgt'], svoFile, pointer_gen)
    # enc_input_extend_vocab: source带有oov的id，oov相对于source_vocab
    # tgt_extend_vocab: tgt带有oov的id，oov相对于tgt_vocab
    # src_oovs_list： source里不再词典里的词
    return train_data, vocab_dicts
