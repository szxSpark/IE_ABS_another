import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, opt, dicts):
        super(Generator, self).__init__()
        self.linear = nn.Linear(opt.dec_rnn_size // opt.maxout_pool_size, dicts['tgt'].size())
        self.sm = nn.Softmax(dim=-1)
        self.pointer_gen = opt.pointer_gen

    def forward(self, g_outputs, g_p_gens, g_attn, enc_batch_extend_vocab, extra_zeros, is_traing=True):
        # g_outputs: (decL-1, B, H)
        # g_p_gens:  (decL-1, B, 1)
        # g_attn (decL-1, B, sourceL)
        # enc_batch_extend_vocab  (sourceL, B)
        # extra_zeros  oovs, B
        # print("enc_batch_extend_vocab.size()", enc_batch_extend_vocab.size())
        if extra_zeros is not None:
            oovs, B = extra_zeros.size()
            # if not is_traing:
            #     print(g_outputs.size(), oovs)

            extra_zeros = extra_zeros.permute(1, 0)  # B, oovs
            B, oovs = extra_zeros.size()
            extra_zeros = extra_zeros.unsqueeze(0).expand(g_outputs.size(0), B, oovs)  # decL-1, B, oovs
            extra_zeros = extra_zeros.contiguous()
            extra_zeros = extra_zeros.view(-1, extra_zeros.size(2))  # decL-1*B, oovs
            # extra_zeros = torch.zeros((g_outputs.size(0)*g_outputs.size(1), oovs)).cuda()  # decL-1*B, oovs

        enc_batch_extend_vocab = enc_batch_extend_vocab.permute(1, 0)  # B, sourceL
        B, sourceL = enc_batch_extend_vocab.size()
        enc_batch_extend_vocab = enc_batch_extend_vocab.unsqueeze(0).expand(g_outputs.size(0), B,
                                                                            sourceL)  # decL-1, B, sourceL
        enc_batch_extend_vocab = enc_batch_extend_vocab.contiguous()
        enc_batch_extend_vocab = enc_batch_extend_vocab.view(-1, enc_batch_extend_vocab.size(2))  # decL-1*B, sourceL

        g_outputs = g_outputs.view(-1, g_outputs.size(2))  # (decL-1*B, H)】

        vocab_dist = self.sm(self.linear(g_outputs))  # (decL-1*B, tgt_size)
        p_gen = g_p_gens.view(-1, 1)  # (decL-1*B, 1)
        attn_dist = g_attn.view(-1, g_attn.size(-1))  # (decL-1*B, sourceL)

        vocab_dist_ = p_gen * vocab_dist  # (decL-1*B, tgt_size)
        attn_dist_ = (1 - p_gen) * attn_dist  # (decL-1*B, sourceL)
        if extra_zeros is not None:
            vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)  # (decL-1*B, tgt_size+oovs)
        final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        return final_dist  # (decL-1*B, tgt_size+oovs)
