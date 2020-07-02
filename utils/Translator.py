import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import utils.Constants as Constants
from models import Encoder, Decoder, DecInit, NMTModel, Beam
from utils.Dataset import Dataset

class Translator(object):
    def __init__(self, opt, model=None, vocab_dicts=None):
        self.opt = opt
        if model is None:
            checkpoint = torch.load(opt.model)
            model_opt = checkpoint['opt']
            self.src_dict = checkpoint['dicts']['src']
            self.tgt_dict = checkpoint['dicts']['tgt']
            self.enc_rnn_size = model_opt.enc_rnn_size
            self.dec_rnn_size = model_opt.dec_rnn_size
            encoder = Encoder(model_opt, self.src_dict)
            decoder = Decoder(model_opt, self.tgt_dict)
            decIniter = DecInit(model_opt)
            model = NMTModel(encoder, decoder, decIniter)

            generator = nn.Sequential(
                nn.Linear(model_opt.dec_rnn_size // model_opt.maxout_pool_size, self.tgt_dict.size()),
                nn.LogSoftmax()
            )

            model.load_state_dict(checkpoint['model'])
            generator.load_state_dict(checkpoint['generator'])

            if opt.cuda:
                model.cuda()
                generator.cuda()
            else:
                model.cpu()
                generator.cpu()

            model.generator = generator
        else:
            self.src_dict = vocab_dicts['src']
            self.tgt_dict = vocab_dicts['tgt']
            self.enc_rnn_size = opt.enc_rnn_size
            self.dec_rnn_size = opt.dec_rnn_size
            self.opt.cuda = True if len(opt.gpus) >= 1 else False
            self.opt.n_best = 1
            self.opt.replace_unk = False  # TODO

        self.tt = torch.cuda if opt.cuda else torch
        self.model = model
        self.model.eval()

        self.copyCount = 0

    def buildData(self, srcBatch, goldBatch, svo_batch):
        srcData = []
        tgtData = [] if goldBatch else None
        svoData = []
        tgt_extend_vocab = [] if goldBatch else None
        src_extend_vocab = []
        src_oovs_list = []
        for i, (srcWords, svo_list) in enumerate(zip(srcBatch, svo_batch)):
            srcData += [self.src_dict.convertToIdx(srcWords,
                                                   Constants.UNK_WORD)]
            svoData += [[self.src_dict.convertToIdx(one_svo, Constants.UNK_WORD) for one_svo in svo_list]]

            if goldBatch:
                tgtData += [self.tgt_dict.convertToIdx(goldBatch[i],
                                                       Constants.UNK_WORD,
                                                       Constants.BOS_WORD,
                                                       Constants.EOS_WORD)]

            if self.opt.pointer_gen:
                # 存储临时的oov词典
                enc_input_extend_vocab, article_oovs = self.article2ids(srcWords, self.src_dict)
                src_extend_vocab += [enc_input_extend_vocab]
                src_oovs_list += [article_oovs]
                if goldBatch:
                    abs_ids_extend_vocab = self.abstract2ids(goldBatch[i], self.tgt_dict, article_oovs)
                    # 覆盖target，用于使用临时词典
                    vec = []
                    vec += [self.src_dict.lookup(Constants.BOS_WORD)]
                    vec += abs_ids_extend_vocab
                    vec += [self.src_dict.lookup(Constants.EOS_WORD)]
                    tgt_extend_vocab.append(torch.LongTensor(vec))

        if goldBatch:
            train = {
                'src': (srcData, svoData),
                'tgt': tgtData,
                'src_extend_vocab': src_extend_vocab,
                'tgt_extend_vocab': tgt_extend_vocab,
                'src_oovs_list': src_oovs_list,
            }
        else:
            train = {
                'src': (srcData, svoData),
                'src_extend_vocab': src_extend_vocab,
                'src_oovs_list': src_oovs_list,
            }
        return Dataset(train, self.opt.batch_size, self.opt.cuda, volatile=True, pointer_gen=self.opt.pointer_gen, is_coverage=self.opt.is_coverage)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, Constants.EOS)
        if tokens[-1] == Constants.EOS_WORD:
            tokens = tokens[:-1]  # EOS
        # if self.opt.replace_unk:
        #     for i in range(len(tokens)):
        #         if tokens[i] == Constants.UNK_WORD:
        #             _, maxIndex = attn[i].max(0)
        #             tokens[i] = src[maxIndex[0]]
        return tokens

    def article2ids(self, article_words, vocab):
        ids = []
        oovs = []
        unk_id = vocab.lookup(Constants.UNK_WORD)
        for w in article_words:
            i = vocab.lookup(w, unk_id)  # 查不到默认unk
            if i == unk_id:  # oov
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                ids.append(vocab.size() + oov_num)
            else:
                ids.append(i)
        return ids, oovs

    def abstract2ids(self, abstract_words, vocab, article_oovs):
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

    def translateBatch(self, srcBatch):

        #   srcBatch: tuple len: 7
        #       srcBatch[0].size()  L, B        source
        #       srcBatch[1].size()  1, B        length, 降序
        #       srcBatch[2].size()  sourceL, B  source_extend_vocab
        #       srcBatch[3].size()  oovs, B     source的最大oovs数目 zeros_expand
        #       srcBatch[4]         list(), len=B, article_oovs
        #       srcBatch[5]         L, B        coverage
        #       srcBatch[6]         L, B        src_sentence_flag_vec

        #   tgtBatch: tuple len: 2
        #       tgtBatch[0].size()  decL, B     target
        #       tgtBatch[1].size()  decL, B     target_extend_vocab
        article_oovs = srcBatch[4]  # len(B)   list(list())

        batchSize = srcBatch[0].size(1)
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(srcBatch)  # L, B , H
        srcBatch_0 = srcBatch[0]

        decStates = self.model.decIniter(encStates[1])  # batch, dec_hidden

        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = Variable(context.data.repeat(1, beamSize, 1))
        decStates = Variable(decStates.unsqueeze(0).data.repeat(1, beamSize, 1))
        decCells = torch.zeros_like(decStates)  # lstm

        att_vec = self.model.make_init_att(context)
        padMask = srcBatch_0.eq(Constants.PAD).transpose(0, 1).unsqueeze(0).repeat(beamSize, 1, 1).float()

        beam = [Beam(beamSize, self.opt.cuda) for k in range(batchSize)]
        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        tmp_src_batch_5 = copy.deepcopy(srcBatch[5])  # L, B
        for i in range(self.opt.max_sent_length):
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).transpose(0, 1).contiguous().view(1, -1)
            # input  (1, B)  # 初始时刻都是BOS, B不是batch，是现在未解码完成的数量
            if srcBatch[5] is not None:
                coverage = torch.stack([tmp_src_batch_5[:, i] for i, b in enumerate(beam) if not b.done],
                                                     dim=1).permute(1,0)  # B, L
            else:
                coverage = None
            if self.opt.pointer_gen:
                # 替换input中不在词典的pointer
                for batch_id in range(input.size(1)):
                    token_id = input[0, batch_id].item()
                    if token_id >= self.tgt_dict.size():
                        oov_id = self.tgt_dict.lookup(Constants.UNK_WORD)  # 用unk代替，获取embedding
                        input[0, batch_id] = oov_id
            # g_outputs, (decStates, decCells), attn, att_vec, g_p_gens, g_attn, _, next_coverage = self.model.decoder(
                # input, (decStates, decCells), context, padMask.view(-1, padMask.size(2)), att_vec, coverage)  # 应该返回coverage
            g_outputs, decStates, attn, att_vec, g_p_gens, g_attn, _, next_coverage = self.model.decoder(
                input, decStates, context, padMask.view(-1, padMask.size(2)), att_vec, coverage)  # 应该返回coverage
            # g_outputs (1, B, H)
            # g_p_gens (1, B, 1)
            # g_attn (1, B, sourceL)
            # next_coverage (B, L)
            if next_coverage is not None:
                count = 0
                for i, b in enumerate(beam):
                    if not b.done:
                        tmp_src_batch_5[:, i] = next_coverage[count, :]
                        count += 1

            if self.opt.pointer_gen:
                enc_batch_extend_vocab = torch.stack([srcBatch[2][:, i] for i, b in enumerate(beam) if not b.done],
                                                     dim=1)
                if self.opt.cuda:
                    enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
                if srcBatch[3] is not None:
                    extra_zeros = torch.stack([srcBatch[3][:, i] for i, b in enumerate(beam) if not b.done], dim=1)
                    if self.opt.cuda:
                        extra_zeros = extra_zeros.cuda()
                else:
                    extra_zeros = None
                # print([i for i, b in enumerate(beam) if not b.done])
                g_out_prob = self.model.generator(g_outputs, g_p_gens, g_attn, enc_batch_extend_vocab,
                                                          extra_zeros, is_traing=False)  # (decL-1*B, tgt_size+oovs)
            else:
                # g_out_prob = self.model.generator(g_outputs, g_p_gens, g_attn, None,
                #                                   None)  # (decL-1*B, tgt_size)
                g_outputs = g_outputs.squeeze(0)
                g_out_prob = self.model.generator.forward(g_outputs)

            # (batch, beam, tgt_size+oovs)  # 如果pointer 需要重新弄
            wordLk = g_out_prob.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            # batch, beamSize, sourceL
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            active = []  # 还需要计算的example
            father_idx = []
            for b in range(batchSize):
                if beam[b].done:
                    continue
                idx = batchIdx[b]
                if not beam[b].advance(wordLk[idx], attn[idx]):  # 有问题
                    active += [b]
                    father_idx.append(beam[b].prevKs[-1])  # this is very annoying
            if not active:
                break
            # to get the real father index
            real_father_idx = []
            for kk, idx in enumerate(father_idx):
                real_father_idx.append(idx * len(father_idx) + kk)  # 每个的原始batchid

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t, rnnSize):
                # select only the remaining active sentences
                # with torch.no_grad():
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents

                # return Variable(view.index_select(1, activeIdx) \
                #                 .view(*newSize), volatile=True)
                return view.index_select(1, activeIdx).view(*newSize)

            decStates = updateActive(decStates, self.dec_rnn_size)
            decCells = updateActive(decCells, self.dec_rnn_size)
            context = updateActive(context, self.enc_rnn_size)
            att_vec = updateActive(att_vec, self.enc_rnn_size)
            # with torch.no_grad():
            # padMask = padMask.index_select(1, Variable(activeIdx, volatile=True))
            padMask = padMask.index_select(1, activeIdx)

            # set correct state for beam search
            previous_index = torch.stack(real_father_idx).transpose(0, 1).contiguous()
            decStates = decStates.view(-1, decStates.size(2)).index_select(0, previous_index.view(-1)).view(
                *decStates.size())
            decCells = decCells.view(-1, decCells.size(2)).index_select(0, previous_index.view(-1)).view(
                *decCells.size())
            att_vec = att_vec.view(-1, att_vec.size(1)).index_select(0, previous_index.view(-1)).view(*att_vec.size())

            remainingSents = len(active)

        # (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch_0.data[:, b].ne(Constants.PAD).nonzero().squeeze(1)
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            # print(len(attn))  # 1

            attn = [a.index_select(1, valid_attn) for a in attn]
            # len(hyps)) 1
            for bb in range(len(hyps[0])):
                token_id = hyps[0][bb].item()
                if token_id >= self.tgt_dict.size():  # 如果不是pointer 不存在这种情况
                    oov = article_oovs[b][token_id - self.tgt_dict.size()]
                    hyps[0][bb] = oov  # str

            allHyp += [hyps]

            allAttn += [attn]
        # allHyp: Batch, Beam, length
        return allHyp, allScores, allAttn, None

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        # (wrap(srcBatch),  lengths), (wrap(tgtBatch), ), indices
        src, tgt, indices = dataset[0]

        #  (2) translate
        pred, predScore, attn, _ = self.translateBatch(src)
        pred, predScore, attn = list(zip(
            *sorted(zip(pred, predScore, attn, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, None
