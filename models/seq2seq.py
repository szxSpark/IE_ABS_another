import torch.nn as nn
import utils.Constants as Constants


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, decIniter):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.encoder_global_context = self.encoder.context_para  # 2*H, 1
        print(id(self.decoder.encoder_global_context))
        print(id(self.encoder.context_para))

        # print("shore encoder decoder gru")
        # self.decoder.rnn.layers[0].weight_ih = self.encoder.rnn.weight_ih_l0
        # self.decoder.rnn.layers[0].weight_hh = self.encoder.rnn.weight_hh_l0
        # self.decoder.rnn.layers[0].bias_ih = self.encoder.rnn.bias_ih_l0
        # self.decoder.rnn.layers[0].bias_hh = self.encoder.rnn.bias_hh_l0
        # print(id(self.decoder.rnn.layers[0].weight_ih),  id(self.encoder.rnn.weight_ih_l0))
        # print(self.decoder.rnn.layers[0].weight_ih is self.encoder.rnn.weight_ih_l0)
        # print(id(decoder.rnn.layers[0].bias_ih), id(self.encoder.rnn.bias_ih_l0))
        # print(decoder.rnn.layers[0].bias_ih is self.encoder.rnn.bias_ih_l0)

        self.decIniter = decIniter

    def make_init_att(self, context):
        # L, B, 2 * H
        batch_size = context.size(1)
        h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return context.data.new(*h_size).zero_()

    def forward(self, input):
        """
        input: (wrap(srcBatch), wrap(srcBioBatch), lengths), (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch))
        """
        # type(input): tuple    len: 2
        #   type(input[0]): tuple len: 6
        #       input[0][0].size()  L, B        source
        #       input[0][1].size()  1, B        length, 降序
        #       input[0][2].size()  L, B        source_extend_vocab
        #       input[0][3].size()  oovs, B     source的最大oovs zeros
        #       input[0][4]         list(), len=B, article_oovs
        #       input[0][5]         L, B        coverage
        #       input[0][6]         L, B        src_sentence_flag_vec

        #   type(input[1]): tuple len: 1
        #       input[1][0].size()  decL, B        target
        #       input[1][1].size()  decL, B        target_extend_vocab
        src = input[0]
        tgt = input[1][0][:-1]  # exclude last target from inputs
        src_pad_mask = src[0].data.eq(Constants.PAD).transpose(0, 1).float()  # B, L
        enc_hidden, context = self.encoder(src)  # (2, B, H) (L, B, 2*H)
        # enc_hidden是rnn的最后一个时间戳
        # context是经过门过滤后的每个词的编码
        init_att = self.make_init_att(context)  # (B, 2*H)  # 都是0
        enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)  # Eq. 11 (1, B，2*H)
        # enc_hidden： 经过linear与tanh， 这么做的好处是，离靠前位置的句子近，不容易丢失信息，符合新闻文章的特点
        coverage = input[0][5]
        if coverage is not None:
            coverage = coverage.permute(1, 0)  # (B, L)
        # g_out, dec_hidden, _attn, _attention_vector, g_p_gens, g_attn, coverage_losses, _ = self.decoder(tgt, (enc_hidden, torch.zeros_like(enc_hidden)), context, src_pad_mask, init_att, coverage)
        g_out, dec_hidden, _attn, _attention_vector, g_p_gens, g_attn, coverage_losses, _ = self.decoder(tgt,
                                                                                                         enc_hidden,
                                                                                                         context,
                                                                                                         src_pad_mask,
                                                                                                         init_att,
                                                                                                         coverage)

        # g_out (decL-1, B, H)
        # g_p_gens (decL-1, B, 1)
        # g_attn (decL-1, B, sourceL)
        #
        return g_out, g_p_gens, g_attn, coverage_losses