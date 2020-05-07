from __future__ import division
import math
import torch
import utils.Constants as Constants
def pad_batch_tensorize(inputs, pad, cuda=True, max_num=0):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    # inputs是一个文章
    # print([len(ids) for ids in inputs])  # 每个句子的单词数
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    try:
        max_len = max(len(ids) for ids in inputs)  # 最大单词数
    except ValueError:
        # print('inputs:', inputs)
        # print('batch_size:', batch_size)
        if inputs == []:
            max_len = 1
            batch_size = 1
    if max_len < max_num:
        max_len = max_num
    # cool !!!
    tensor_shape = (batch_size, max_len)  # max_len >= max_num
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    # sent_num, word_num, 1个文章
    return tensor


class Dataset(object):
    def __init__(self, dataset, batchSize, cuda, volatile=False, pointer_gen=False, is_coverage=False):
        assert type(dataset) == dict
        srcData, svoData = dataset['src']
        if 'tgt' in dataset:
            tgtData = dataset['tgt']
        else:
            tgtData = None

        self.src = srcData
        self.svo = svoData
        assert len(self.src) == len(self.svo)

        if tgtData:
            self.tgt = tgtData
            assert len(self.src) == len(self.tgt)
            assert len(self.svo) == len(self.tgt)

        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)
        self.volatile = volatile

        self.pointer_gen = pointer_gen
        self.is_coverage = is_coverage
        if self.pointer_gen:
            self.src_extend_vocab = dataset['src_extend_vocab']
            self.tgt_extend_vocab = dataset['tgt_extend_vocab']
            self.src_oovs_list = dataset['src_oovs_list']

    def _batchify(self, data, start_idx, end_idx, align_right=False, include_lengths=False, is_src=True):

        # source = pad_batch_tensorize(sources, pad=Constants.PAD, cuda=self.cuda)
        # tar_in = pad_batch_tensorize(tar_ins, pad=Constants.PAD, cuda=self.cuda)
        # target = pad_batch_tensorize(targets, pad=Constants.PAD, cuda=self.cuda)  # ext
        # ext_src = pad_batch_tensorize(ext_srcs, pad=Constants.PAD, cuda=self.cuda)  # ext
        #
        # sources = [pad_batch_tensorize(inputs=_, pad=Constants.PAD, cuda=self.cuda, max_num=5) for _ in
        #            source_lists]  # 每个元素，sent_num, word_num, 1个文章
        # id_svos = [pad_batch_tensorize(inputs=_, pad=Constants.PAD, cuda=self.cuda, max_num=4) for _ in
        #            id_svos_list]  # 每个元素，len(svo), word_num, 1个文章
        if is_src:
            src_data, svo_data = data
            data = src_data
            source = pad_batch_tensorize(data, pad=Constants.PAD, cuda=self.cuda)
            print(source.size())
            # id_svos = [pad_batch_tensorize(inputs=_, pad=Constants.PAD, cuda=self.cuda, max_num=4) for _ in
            #            svo_data]  # 每个元素，len(svo), word_num, 1个文章
            # for a in id_svos:
            #     print(a.size())  #11
            # input()


        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        enc_batch_extend_vocab = None
        dec_batch_extend_vocab = None
        extra_zeros = None
        article_oovs = None
        coverage = None

        if self.pointer_gen:
            if is_src:
                article_oovs = self.src_oovs_list[start_idx:end_idx]
                max_art_oovs = max([len(_article_oovs) for _article_oovs in article_oovs])  # 最长的oovs长度
                src_extend_vocab = self.src_extend_vocab[start_idx:end_idx]
                enc_batch_extend_vocab = data[0].new(len(data), max_length).fill_(Constants.PAD)
                for i in range(len(data)):
                    data_length = data[i].size(0)
                    offset = max_length - data_length if align_right else 0
                    enc_batch_extend_vocab[i].narrow(0, offset, data_length).copy_(torch.LongTensor(src_extend_vocab[i]))
                if max_art_oovs > 0:
                    extra_zeros = torch.zeros((self.batchSize, max_art_oovs))
            else:  # tgt
                tgt_extend_vocab = self.tgt_extend_vocab[start_idx:end_idx]
                dec_batch_extend_vocab = data[0].new(len(data), max_length).fill_(Constants.PAD)
                for i in range(len(data)):
                    data_length = data[i].size(0)
                    offset = max_length - data_length if align_right else 0
                    dec_batch_extend_vocab[i].narrow(0, offset, data_length).copy_(
                        torch.LongTensor(tgt_extend_vocab[i]))

        if self.is_coverage:
            coverage = torch.zeros(len(data), max_length)

        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        print(out.size())
        print(out)
        print(source)
        input()

        # out, enc_batch_extend_vocab 的 size相同
        if include_lengths and is_src:  # src
            return out, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage, lengths
        if not include_lengths and not is_src:  # tgt
            return out, dec_batch_extend_vocab

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)

        start_idx = index * self.batchSize
        end_idx = (index + 1) * self.batchSize
        srcBatch, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage, lengths = self._batchify(
            [self.src[start_idx:end_idx], self.svo[start_idx:end_idx]],
            start_idx=index * self.batchSize,
            end_idx=(index + 1) * self.batchSize,
            align_right=False, include_lengths=True, is_src=True)
        if self.tgt:
            tgtBatch, dec_batch_extend_vocab = self._batchify(
                self.tgt[start_idx:end_idx],
                start_idx=index * self.batchSize,
                end_idx=(index + 1) * self.batchSize,
                is_src=False
            )
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None:
            if self.pointer_gen:
                if extra_zeros is not None:
                    batch = zip(indices, srcBatch, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage)
                else:
                    batch = zip(indices, srcBatch, enc_batch_extend_vocab, article_oovs, coverage, )
            else:
                batch = zip(indices, srcBatch, )
        else:
            if self.pointer_gen:
                if extra_zeros is not None:
                    batch = zip(indices, srcBatch, tgtBatch, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage, dec_batch_extend_vocab)
                else:
                    batch = zip(indices, srcBatch, tgtBatch, enc_batch_extend_vocab, article_oovs, coverage, dec_batch_extend_vocab)
            else:
                batch = zip(indices, srcBatch, tgtBatch,)

        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            if self.pointer_gen:
                if extra_zeros is not None:
                    indices, srcBatch, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage, = zip(*batch)
                else:
                    indices, srcBatch, enc_batch_extend_vocab, article_oovs, coverage, = zip(*batch)
                    extra_zeros = None
            else:
                indices, srcBatch,  = zip(*batch)
        else:
            if self.pointer_gen:
                if extra_zeros is not None:
                    indices, srcBatch, tgtBatch, enc_batch_extend_vocab, extra_zeros, article_oovs, coverage, dec_batch_extend_vocab = zip(*batch)
                else:
                    indices, srcBatch, tgtBatch, enc_batch_extend_vocab, article_oovs, coverage, dec_batch_extend_vocab = zip(*batch)
                    extra_zeros = None
            else:
                indices, srcBatch, tgtBatch, = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            return b

        lengths = torch.LongTensor(lengths).view(1, -1)
        return (wrap(srcBatch), lengths, wrap(enc_batch_extend_vocab), wrap(extra_zeros), article_oovs, wrap(coverage)), \
               (wrap(tgtBatch), wrap(dec_batch_extend_vocab)), \
               indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        if self.pointer_gen:
            data = list(zip(self.src, self.tgt, self.src_extend_vocab, self.tgt_extend_vocab, self.src_oovs_list))
            self.src, self.tgt, self.src_extend_vocab, self.tgt_extend_vocab, self.src_oovs_list = zip(*[data[i] for i in torch.randperm(len(data))])
        else:
            data = list(zip(self.src, self.tgt))
            self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
