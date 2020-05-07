import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.Constants as Constants
import models.modules


class Decoder(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed

        input_size = opt.word_vec_size

        super(Decoder, self).__init__()
        if opt.share_embedding:
            self.word_lut = None
        else:
            self.word_lut = nn.Embedding(dicts.size(),
                                         input_size,
                                         padding_idx=Constants.PAD)
        if self.input_feed:
            input_size += opt.enc_rnn_size

        self.rnn_type = "gru"
        if self.rnn_type == "gru":
            self.rnn = models.modules.StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        else:
            self.rnn = models.modules.StackedLSTM(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)

        # self.attn = s2s.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.attn = models.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size + opt.enc_rnn_size, opt.att_vec_size)
        self.dropout = nn.Dropout(opt.dropout)

        # self.readout = nn.Linear((opt.enc_rnn_size + opt.dec_rnn_size + opt.word_vec_size), opt.dec_rnn_size)
        self.readout = nn.Linear((opt.enc_rnn_size + opt.dec_rnn_size + opt.word_vec_size + opt.enc_rnn_size), opt.dec_rnn_size)

        self.maxout = models.modules.MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size

        self.hidden_size = opt.dec_rnn_size
        self.pointer_gen = opt.pointer_gen
        self.is_coverage = opt.is_coverage
        if self.pointer_gen:
            self.p_gen_linear = nn.Linear(self.hidden_size * 2 + opt.word_vec_size, 1)
        self.encoder_global_context = None

    def load_lm_rnn(self, opt):
        if opt.lm_model_file is not None:
            print("load lm decoder ...")
            para_dict = torch.load(opt.lm_model_file)

            def rename_para_key(original_key, new_key):
                para_dict[new_key] = para_dict[original_key]
                del para_dict[original_key]

            rename_para_key('weight_ih_l0', 'layers.0.weight_ih')
            rename_para_key('weight_hh_l0', 'layers.0.weight_hh')
            rename_para_key('bias_ih_l0', 'layers.0.bias_ih')
            rename_para_key('bias_hh_l0', 'layers.0.bias_hh')

            missing_keys, unexpected_keys = self.rnn.load_state_dict(para_dict)
            print("missing_keys:", missing_keys)
            print("unexpected_keys:", unexpected_keys)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            print("Loading pretrain embedding...")
            pretrained_weight = np.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def forward(self, input_y, hidden, context, src_pad_mask, init_att, coverage):
        # input: (decL-1, B)    target
        # hidden: (1, B，2*H)   经过linear与tanh得到的初始hidden
        # context: (L, B, 2*H)  encoder是经过门过滤后的每个词的编码
        # src_pad_mask: (B, L)  pad的是1
        # init_att: (B, 2*H)    zero
        # coverage: (B, L)
        emb = self.word_lut(input_y)  # (decL-1, B, word_H)
        g_outputs = []
        g_p_gens = []
        g_attn = []
        cur_context = init_att  # (B, 2*H)
        # print(self.attn)  ConcatAttention(512 * (512->512 + 512->512))
        self.attn.applyMask(src_pad_mask)
        precompute = None
        coverage_loss_list = []
        coverage_losses = None
        for step, emb_t in enumerate(emb.split(1)):  # 分解每个长度, 根据第1个维度，即decL-1
            emb_t = emb_t.squeeze(0)  # (B, word_H)
            input_emb = emb_t  # (B, word_H)
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context], 1)  # 利用att  (B, H+word_H)
            output, hidden = self.rnn(input_emb, hidden)

            # output是最后，hidden是num_layers cat 在一起的，用于传入下一个阶段
            # output: (B, 2*H)
            # hidden: (1, B, 2*H)
            cur_context, attn, precompute, next_coverage = self.attn(torch.cat((output,
                                                                                self.encoder_global_context.squeeze().unsqueeze(0).expand_as(output),
                                                                     ), dim=1),
                                                                     context.transpose(0, 1), precompute, coverage)
            # cur_context: (B, 2*H), 加权求和后的上下文
            # attn: (B, L)  sourceL,  score
            # precompute: (B, L, 2*H)
            # next_coverage: (B, L) 已经加上attn
            # ---------------------------------------------
            if self.pointer_gen:
                # cur_context, output, emb_t(or input_emb),
                p_gen_input = torch.cat((cur_context, output, emb_t), 1)  # B, (2*2*H + word_H)
                p_gen = self.p_gen_linear(p_gen_input)  # B, 1
                p_gen = F.sigmoid(p_gen)  # B, 1
                g_p_gens += [p_gen]
                g_attn += [attn]
            # ---------------------------------------------
            if self.encoder_global_context is None:
                readout = self.readout(torch.cat((emb_t, output, cur_context), dim=1))  # B, 2*H
            else:
                readout = self.readout(torch.cat((emb_t, output, cur_context,
                                                  self.encoder_global_context.squeeze().unsqueeze(0).expand_as(cur_context)
                                                  ), dim=1))  # B, 2*H 或许可以在搞搞，比如拿来做attention
            maxout = self.maxout(readout)  # B, H
            output = self.dropout(maxout)  # B, H
            g_outputs += [output]

            if self.is_coverage:
                # coverage 未加attn，t-1时刻钱的累加和
                step_coverage_loss = torch.sum(torch.min(attn, coverage), 1)  # B
                step_coverage_loss = self.opt.cov_loss_wt * step_coverage_loss  # B
                coverage_loss_list.append(step_coverage_loss)
                coverage = next_coverage

        g_outputs = torch.stack(g_outputs)  # (decL-1, B, H)
        if self.pointer_gen:
            g_p_gens = torch.stack(g_p_gens)  # (decL-1, B, 1)
            g_attn = torch.stack(g_attn)  # (decL-1, B, sourceL)
        if self.is_coverage:
            coverage_losses = torch.stack(coverage_loss_list, dim=0)  # decL-1, B
        return g_outputs, hidden, attn, cur_context, g_p_gens, g_attn, coverage_losses, next_coverage
