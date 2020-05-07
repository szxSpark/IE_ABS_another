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
            if isinstance(input, tuple):
                outputs = unpack(outputs)[0]
        else:
            outputs, (hidden_t, _) = self.rnn(emb)
            if isinstance(input, tuple):
                outputs = unpack(outputs)[0]

        time_step = outputs.size(0)  # L
        batch_size = outputs.size(1)  # B

        outputs = outputs.permute(1, 0, 2)  # B, L, 2*H
        u = F.tanh(self.context_proj(outputs))  # B, L, 2*H
        attention = torch.matmul(u, self.context_para).squeeze()  # B,L,2*H   2*H, 1
        attention = F.softmax(attention, dim=0)  # B, L
        sentence_vector = torch.bmm(attention.unsqueeze(dim=1), outputs).squeeze(dim=1)  # B, 2*H
        outputs = outputs.permute(1, 0, 2)  # L, B, 2*H

        hidden_t = (None, sentence_vector)
        return hidden_t, outputs
