import torch
import torch.nn as nn
import utils.Constants as Constants
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F
import numpy as np

class ConvEntityEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, emb_dim, n_hidden, dropout, embedding=None):
        super().__init__()
        self._embedding = embedding
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(2, 5)])
        self._dropout = dropout

    def forward(self, input_):
        # print(input_)  # 实体类数量，长度，（一个文章）
        emb_input = self._embedding(input_)
        # print(emb_input.size())  # cluster_n, L, H
        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        # cluster_n, 3*n_hidden
        return output

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
        # self.svo_para = nn.Parameter(torch.FloatTensor(300, 1))
        self.svo_proj = nn.Linear(300, 300, bias=True)
        self.svo_query_proj = nn.Linear(self.hidden_size * 2, 300, bias=True)

        # self.selective_gate = nn.Linear(self.hidden_size * 2 * 2, self.hidden_size * 2)
        # self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(opt.dropout)

        conv_hidden = 100
        self._entity_enc = ConvEntityEncoder(
            opt.word_vec_size, conv_hidden, opt.dropout, embedding=self.word_lut,
        )
        self.context_proj = nn.Linear(self.hidden_size * 2, 300, bias=True)
        self.szx_proj = nn.Linear(300,self.hidden_size * 2, bias=True)



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

    def _encode_entity(self, clusters):
        cluster_nums = [c.size(0) for c in clusters]
        enc_entities = [self._entity_enc(cluster_words) for cluster_words in clusters]
        max_n = max(cluster_nums)
        def zero(n, device):
            z = torch.zeros(n, 300).to(device)
            return z
        enc_entity = torch.stack(
            [torch.cat([s, zero(max_n - n, s.device)], dim=0)
             if n != max_n
             else s
             for s, n in zip(enc_entities, cluster_nums)],
            dim=0
        )
        # [batch_size, max_n，3*n_hidden]
        return enc_entity

    def forward(self, input, id_hidden=None):
        # type(input): tuple    len: 7
        #   input[0].size()  L, B        source
        #   input[1].size()  1, B        length, 降序
        svo_list = input[1]
        lengths = input[2].data.view(-1).tolist()
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

        # ------ svo
        svo_query = outputs.permute(1, 0, 2).max(dim=1)[0]  # B, 2*H
        svo_query = self.svo_query_proj(svo_query).unsqueeze(dim=2)  # B, 300, 1
        entity_out = self._encode_entity(svo_list)  # B, entity_num, 300
        u = F.tanh(self.svo_proj(entity_out))  # B, entity_num, 300
        entity_attention = torch.bmm(u, svo_query).squeeze(dim=-1)  # B, entity_num
        # entity_attention = torch.matmul(u, self.svo_para).squeeze(dim=-1)  # B,entity_num,300   300, 1
        entity_attention = F.softmax(entity_attention, dim=1)  # B, entity_num
        entity_aware_vector = torch.bmm(entity_attention.unsqueeze(dim=1), entity_out).squeeze(dim=1)  # B, 300
        # ------ svo

        # -----
        # 下边的目的，单词+句子
        # ------ begin han attention
        outputs = outputs.permute(1, 0, 2)  # B, L, 2*H
        u = F.tanh(self.context_proj(outputs))  # B, L, 300
        attention = torch.bmm(u, entity_aware_vector.unsqueeze(dim=2)).squeeze(dim=-1)  # B,L
        attention = F.softmax(attention, dim=1)  # B, L  # TODO
        sentence_vector = torch.bmm(attention.unsqueeze(dim=1), outputs).squeeze(dim=1)  # B, 2*H


        entity_aware_vector = self.szx_proj(entity_aware_vector)  # B, 2*H
        outputs = outputs+entity_aware_vector.unsqueeze(1)  # B, L, 2*H

        outputs = outputs.permute(1, 0, 2)  # L, B, 2*H
        # outputs = self.dropout(outputs)  # dropout

        # ------ end han attention




        hidden_t = (None, sentence_vector)

        # # ------ begin original
        # # sentence_vector = torch.cat((forward_last, backward_last), dim=1)  # B, 2*H
        # # B, 4*H
        # exp_buf = torch.cat((outputs,
        #                      sentence_vector.unsqueeze(0).expand_as(outputs)), dim=2)  # L, B, 4*H
        # selective_value = self.sigmoid(self.selective_gate(exp_buf.view(-1, exp_buf.size(2))))  # Eq.8
        # selective_value = selective_value.view(time_step, batch_size, -1)  # L, B, 2*H
        # outputs = outputs * selective_value  # L, B, 2*H
        # # ------ end original

        # ------ begin another gate
        # selective_value_context = self.sigmoid(self.selective_gate_context(exp_buf.view(-1, exp_buf.size(2))))
        # selective_value_context = selective_value_context.view(time_step, batch_size, -1)  # L, B, 2*H
        # outputs = outputs * selective_value + sentence_vector * selective_value_context  # L, B, 2*H
        # ------ end another gate

        return hidden_t, outputs
        # (2, B, H) (L, B, 2*H)
