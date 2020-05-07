import torch
import torch.nn as nn
import utils.Constants as Constants
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2
        assert opt.enc_rnn_size % self.num_directions == 0
        self.hidden_size = opt.enc_rnn_size // self.num_directions
        input_size = opt.word_vec_size
        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=Constants.PAD)
        self.rnn_type = "gru"
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size, self.hidden_size,
                              num_layers=1,
                              dropout=opt.dropout,
                              bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_size, self.hidden_size,
                               num_layers=opt.layers,
                               dropout=opt.dropout,
                               bidirectional=True)
        self.context_para = nn.Parameter(torch.FloatTensor(self.hidden_size * 2, 1))
        self.context_proj = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.selective_gate = nn.Linear(self.hidden_size * 2 * 3, self.hidden_size * 2)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(opt.dropout)

    def load_lm_rnn(self, opt):
        if opt.lm_model_file is not None:
            print("load lm encoder ...")
            para_dict = torch.load(opt.lm_model_file)

            missing_keys, unexpected_keys = self.word_lut.load_state_dict({'weight': para_dict['weight']})
            print("missing_keys:", missing_keys)
            print("unexpected_keys:", unexpected_keys)
            del para_dict['weight']
            missing_keys, unexpected_keys = self.rnn.load_state_dict(para_dict)
            print("missing_keys:", missing_keys)
            print("unexpected_keys:", unexpected_keys)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            print("Loading pretrain embedding...")
            pretrained_weight = np.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def forward(self, input, hidden=None):
        # type(input): tuple    len: 7
        #   input[0].size()  L, B        source
        #   input[1].size()  1, B        length, 降序
        #   input[6].size()  L, B        src_sentence_flag_vec

        lengths = input[1].data.view(-1).tolist()
        wordEmb = self.word_lut(input[0])  # L, B, H

        emb = pack(wordEmb, lengths)
        if self.rnn_type == "gru":
            outputs, hidden_t = self.rnn(emb)
            forward_last = hidden_t[0]  # B, H
            backward_last = hidden_t[1]  # B, H
            if isinstance(input, tuple):
                outputs = unpack(outputs)[0]
            # h_n (num_layers * num_directions, batch, hidden_size)
        else:
            outputs, (hidden_t, _) = self.rnn(emb)
            forward_last = hidden_t[0]  # B, H
            backward_last = hidden_t[1]  # B, H
            # h_n (num_layers * num_directions, batch, hidden_size)
            if isinstance(input, tuple):
                outputs = unpack(outputs)[0]

        time_step = outputs.size(0)  # L
        batch_size = outputs.size(1)  # B

        # outputs = self.dropout(outputs)  # dropout

        # -----
        # 下边的目的，单词+句子
        # ------ begin han attention
        outputs = outputs.permute(1, 0, 2)  # B, L, 2*H
        u = F.tanh(self.context_proj(outputs))  # B, L, 2*H
        # attention = context_para(u).squeeze()  # B, L
        attention = torch.matmul(u, self.context_para).squeeze()  # B,L,2*H   2*H, 1
        attention = F.softmax(attention, dim=0)  # B, L
        sentence_vector = torch.bmm(attention.unsqueeze(dim=1), outputs).squeeze(dim=1)  # B, 2*H
        outputs = outputs.permute(1, 0, 2)  # L, B, 2*H
        # outputs = self.dropout(outputs)  # dropout

        # ------ end han attention

        # ------ begin self atttntion
        # outputs = self.self_attn(outputs, outputs, outputs, key_padding_mask=src_pad_mask)[0]  # L, B, 2*H
        # # mead pool or max pool
        # sentence_vector = torch.mean(outputs, dim=0)  # B, 2*H
        # ------ end self atttntion

        hidden_t = (None, sentence_vector)

        # ------ begin original
        # sentence_vector = torch.cat((forward_last, backward_last), dim=1)  # B, 2*H
        # B, 4*H
        exp_buf = torch.cat((outputs,
                             self.context_para.squeeze().unsqueeze(0).expand_as(outputs),
                             sentence_vector.unsqueeze(0).expand_as(outputs)), dim=2)  # L, B, 6*H
        selective_value = self.sigmoid(self.selective_gate(exp_buf.view(-1, exp_buf.size(2))))  # Eq.8
        selective_value = selective_value.view(time_step, batch_size, -1)  # L, B, 2*H
        outputs = outputs * selective_value  # L, B, 2*H
        # ------ end original

        # ------ begin another gate
        # selective_value_context = self.sigmoid(self.selective_gate_context(exp_buf.view(-1, exp_buf.size(2))))
        # selective_value_context = selective_value_context.view(time_step, batch_size, -1)  # L, B, 2*H
        # outputs = outputs * selective_value + sentence_vector * selective_value_context  # L, B, 2*H
        # ------ end another gate

        return hidden_t, outputs
        # (2, B, H) (L, B, 2*H)
