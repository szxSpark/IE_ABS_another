from __future__ import division
import argparse
import torch
import torch.nn as nn
from models import Optim, NMTModel, Encoder, Decoder, DecInit, Generator
from torch import cuda
import math
import time
import logging
import os
from PyRouge.Rouge import Rouge
import xargs
from tqdm import tqdm
from utils import onlinePreprocess
from utils.onlinePreprocess import prepare_data_online
import utils.Constants as Constants
from utils.Translator import Translator
from utils.Dataset import Dataset
import json
parser = argparse.ArgumentParser(description='train.py')
xargs.add_data_options(parser)
xargs.add_model_options(parser)
xargs.add_train_options(parser)
opt = parser.parse_args()

logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
if opt.log_home:
    log_file_name = os.path.join(opt.log_home, log_file_name)
file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-'
                                            '5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)
logger.info('My PID is {0}'.format(os.getpid()))
logger.info(opt)

if torch.cuda.is_available() and not opt.gpus:
    logger.info("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.seed > 0:
    torch.manual_seed(opt.seed)

if opt.gpus:
    if opt.cuda_seed > 0:
        torch.cuda.manual_seed(opt.cuda_seed)
    cuda.set_device(opt.gpus[0])

logger.info('My seed is {0}'.format(torch.initial_seed()))
logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))

def load_train_data():
    onlinePreprocess.seq_length = opt.max_sent_length_source  # 训练的截断
    onlinePreprocess.shuffle = 1 if opt.process_shuffle else 0
    train_data, vocab_dicts = prepare_data_online(opt)
    trainData = Dataset(train_data, opt.batch_size, opt.gpus, pointer_gen=opt.pointer_gen, is_coverage=opt.is_coverage)
    logger.info(' * vocabulary size. source = %d; target = %d' %
                (vocab_dicts['src'].size(), vocab_dicts['tgt'].size()))
    logger.info(' * number of training sentences. %d' %
                len(train_data['src']))
    return trainData, vocab_dicts

def load_dev_data(translator, src_file, tgt_file, svo_File):
    def addPair(f1, f2, f3):
        for x, y1, e in zip(f1, f2, f3):
            yield (x, y1, e)
        yield (None, None, None)
    dataset, raw = [], []
    srcF = open(src_file, encoding='utf-8')
    tgtF = open(tgt_file, encoding='utf-8')
    svoF = open(svo_File, encoding='utf-8')

    src_batch, tgt_batch, svo_batch = [], [], []
    for line, tgt, svoline in addPair(srcF, tgtF, svoF):
        if (line is not None) and (tgt is not None) and (svoline is not None) :
            src_tokens = line.strip().split(' ')
            src_tokens = src_tokens[:2000]  # TODO
            src_batch += [src_tokens]
            tgt_tokens = tgt.strip().split(' ')
            tgt_batch += [tgt_tokens]
            svo_list = [one_svo.split(" ") for one_svo in json.loads(svoline, encoding="utf-8")]
            svo_batch += [svo_list]
            if len(src_batch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(src_batch) == 0:
                break
        data = translator.buildData(src_batch, tgt_batch, svo_batch)
        dataset.append(data)
        raw.append((src_batch, tgt_batch))
        src_batch, tgt_batch, svo_batch = [], [], []
    srcF.close()
    tgtF.close()
    return (dataset, raw)

def bulid_model(vocab_dicts):
    logger.info(' * maximum batch size. %d' % opt.batch_size)
    logger.info('Building model...')

    encoder = Encoder(opt, vocab_dicts['src'])
    decoder = Decoder(opt, vocab_dicts['tgt'])
    if opt.share_embedding:
        decoder.word_lut = encoder.word_lut
    decIniter = DecInit(opt)
    model = NMTModel(encoder, decoder, decIniter)

    if opt.pointer_gen:
        generator = Generator(opt, vocab_dicts)  # TODO 考虑加dropout
    else:
        generator = nn.Sequential(
            nn.Linear(opt.dec_rnn_size // opt.maxout_pool_size, vocab_dicts['tgt'].size()),
            # nn.Linear(opt.word_vec_size, dicts['tgt'].size()),  # transformer
            nn.LogSoftmax(dim=-1)
        )
    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()
    model.generator = generator
    logger.info("model.encoder.word_lut: {}".format(id(model.encoder.word_lut)))
    logger.info("model.decoder.word_lut: {}".format(id(model.decoder.word_lut)))
    logger.info("embedding share: {}".format(model.encoder.word_lut is model.decoder.word_lut))
    logger.info(repr(model))
    param_count = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('total number of parameters: %d\n\n' % param_count)

    init_params(model)
    optim = build_optim(model)

    # # 断点重连
    # logger.info(opt.checkpoint_file)
    #
    # if opt.checkpoint_file is not None:
    #     logger.info("load {}".format(opt.checkpoint_file))
    #     checkpoint = torch.load(opt.checkpoint_file)
    #     for k, v in checkpoint['generator'].items():
    #         checkpoint['model']["generator."+k] = v
    #     model.load_state_dict(checkpoint['model'])
    #     optim = checkpoint['optim']
    #     opt.start_epoch += checkpoint['epoch']

    return model, optim

def init_params(model):
    logger.info("xavier_normal init")
    for pr_name, p in model.named_parameters():
        logger.info(pr_name)
        if p.dim() == 1:
            p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
        else:
            nn.init.xavier_normal_(p, math.sqrt(3))
    model.encoder.load_pretrained_vectors(opt)
    model.decoder.load_pretrained_vectors(opt)
    # logger.info("load lm rnn")
    # encoder.load_lm_rnn(opt)

def build_optim(model):
    optim = Optim(
        opt.optim, opt.learning_rate,
        max_grad_norm=opt.max_grad_norm,
        min_lr=opt.min_lr,
        max_weight_value=opt.max_weight_value,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        decay_bad_count=opt.halve_lr_bad_count
    )
    # ---- 不同学习率
    small_lr_layers_id = list(map(id, model.encoder.word_lut.parameters())) + \
                         list(map(id, model.encoder.rnn.parameters()))
    large_lr_layers = list(filter(lambda p: id(p) not in small_lr_layers_id, model.parameters()))
    small_lr_layers = list(filter(lambda p: id(p) in small_lr_layers_id, model.parameters()))
    params = [
        {"params": large_lr_layers},
        {"params": small_lr_layers, "lr": 1e-3}
    ]
    # params = model.parameters()
    optim.set_parameters(params, model.parameters())
    return optim

def build_loss(vocabSize):
    weight = torch.ones(vocabSize)
    weight[Constants.PAD] = 0
    loss = nn.NLLLoss(weight, reduction='sum')
    if opt.gpus:
        loss.cuda()
    return loss

def compute_loss(g_outputs, g_targets, generator, crit, g_p_gens, g_attn, enc_batch_extend_vocab, extra_zeros, coverage_losses):
    # g_outputs: (decL-1, B, H)
    # g_targets: (decL-1, B)
    # g_p_gens:  (decL-1, B, 1)
    # coverage_losses  (decL-1, B)
    if opt.pointer_gen:
        dec_padding_mask = 1 - g_targets.eq(Constants.PAD)  # (decL-1, B)
        dec_padding_mask = dec_padding_mask.view(-1).float()  # (decL-1*B,)
        g_prob_t = generator(g_outputs, g_p_gens, g_attn, enc_batch_extend_vocab, extra_zeros)  # (decL-1*B, tgt_size)
        target = g_targets.view(-1)  # (decL-1*B,)
        gold_probs = torch.gather(g_prob_t, 1, target.unsqueeze(1)).squeeze()  # (decL-1*B,)  目的是拿到金的
        step_loss = -torch.log(gold_probs + 1e-12)  # (decL-1*B,)  generator应该是sm
        nll_loss = torch.sum(step_loss * dec_padding_mask).item()
        if opt.is_coverage:
            step_loss = step_loss + coverage_losses.view(-1)  # (decL-1*B,)
        step_loss = step_loss * dec_padding_mask  # (decL-1*B,)
        total_loss = torch.sum(step_loss)
        report_loss = total_loss.item()
        if opt.is_coverage:
            coverage_report_loss = torch.sum(coverage_losses.view(-1)*dec_padding_mask).item()
            return total_loss, (report_loss, nll_loss, coverage_report_loss), 0
        return total_loss, report_loss, 0
    else:
        g_out_t = g_outputs.view(-1, g_outputs.size(2))
        g_prob_t = generator(g_out_t)
        g_loss = crit(g_prob_t, g_targets.view(-1))
        total_loss = g_loss
        report_loss = total_loss.item()
        return total_loss, report_loss, 0

def save_model(model, optim, vocab_dicts, epoch, metric=None):
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
    generator_state_dict = model.generator.module.state_dict() if len(
        opt.gpus) > 1 else model.generator.state_dict()
    checkpoint = {
        'model': model_state_dict,
        'generator': generator_state_dict,
        'dicts': vocab_dicts,
        'opt': opt,
        'epoch': epoch,
        'optim': optim
    }
    save_model_path = 'model'
    if opt.save_path:
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        save_model_path = opt.save_path + os.path.sep + save_model_path
    if metric is not None:
        torch.save(checkpoint,
                   '{0}_devRouge_{1}_{2}_e{3}.pt'.format(save_model_path, round(metric[0], 4), round(metric[1], 4),
                                                         epoch))
    else:
        torch.save(checkpoint, '{0}_e{1}.pt'.format(save_model_path, epoch))

evalModelCount = 0
totalBatchCount = 0
rouge_calculator = Rouge.Rouge()

def evalModel(translator, evalData):
    global evalModelCount
    global rouge_calculator
    evalModelCount += 1
    ofn = 'dev.out.{0}.txt'.format(evalModelCount)
    if opt.save_path:
        ofn = os.path.join(opt.save_path, ofn)
    predict, gold = [], []
    processed_data, raw_data = evalData
    # （list(Dataset)， list((src_batch, tgt_batch)))
    for batch, raw_batch in tqdm(zip(processed_data, raw_data)):
        # (wrap(srcBatch), lengths), (wrap(tgtBatch), ), indices

        src, tgt, indices = batch[0]
        # print(len(src))  # 5
        # print(len(tgt))  # 2
        src_batch, tgt_batch = raw_batch

        #  (2) translate
        with torch.no_grad():
            pred, predScore, attn, _ = translator.translateBatch(src)  # 无teacher forcing
            pred, predScore, attn = list(zip(
                *sorted(zip(pred, predScore, attn, indices),
                        key=lambda x: x[-1])))[:-1]
        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):  # B
            n = 0
            predBatch.append(
                translator.buildTargetTokens(pred[b][n], src_batch[b], attn[b][n])
            )
        gold += [' '.join(r) for r in tgt_batch]
        predict += [' '.join(sents) for sents in predBatch]

    if opt.subword:
        with open(ofn + ".tmp", 'w', encoding='utf-8') as of:
            for p in predict:
                of.write(p + '\n')
        # ofn: pred_file
        cmd = "sed -r 's/(@@ )|(@@ ?$)//g' {} > {}".format(ofn + ".tmp", ofn)  # 写的是分词的
        os.system(cmd)
        os.remove(ofn + ".tmp")
        predict = []
        # opt.dev_ref 不要用subword
        with open(ofn, encoding='utf-8') as f:
            for line in f:
                if not line:
                    break
                # predict.append(line.strip())
                tmp_line = " ".join(list("".join(line.strip().split(" "))))
                tmp_line = tmp_line.replace("< u n k >", "<unk>")
                predict.append(tmp_line)  # 按照char计算指标
        assert len(predict) == len(gold)
        print("gold:", gold[0])
        print("predict:", predict[0])
        scores = rouge_calculator.compute_rouge(gold, predict)
        return scores['rouge-1']['f'][0], scores['rouge-2']['f'][0]
    else:
        if not opt.english:
            for i in range(len(predict)):
                tmp_line = predict[i]
                tmp_line = " ".join(list("".join(predict[i].strip().split(" "))))
                tmp_line = tmp_line.replace("< u n k >", "<unk>")
                predict[i] = tmp_line
        assert len(predict) == len(gold)
        print("gold:", gold[0])
        print("predict:", predict[0])
        scores = rouge_calculator.compute_rouge(gold, predict)
        with open(ofn, 'w', encoding='utf-8') as of:
            for p in predict:
                of.write(p + '\n')
        return scores['rouge-1']['f'][0], scores['rouge-2']['f'][0]

def trainModel(model, translator, trainData, validData, vocab_dicts, optim):
    model.train()
    logger.warning("Set model to {0} mode".format('train' if model.training else 'eval'))
    criterion = build_loss(vocab_dicts['tgt'].size())

    def trainEpoch(epoch):
        if opt.extra_shuffle and epoch > opt.curriculum:
            logger.info('Shuffling...')
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))
        total_loss, total_words = 0, 0
        if opt.is_coverage:
            total_nll_loss, total_coverage_report_loss = 0, 0
        report_loss = 0
        start = time.time()
        for i in range(len(trainData)):
            global totalBatchCount
            totalBatchCount += 1
            # (wrap(srcBatch), lengths), (wrap(tgtBatch)), indices
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1]  # exclude original indices
            # type(batch): tuple    len: 2
            #   type(batch[0]): tuple len: 7
            #       batch[0][0].size()  L, B        source
            #       batch[0][1].size()  list        svo
            #       batch[0][2].size()  1, B        length, 降序
            #       batch[0][3].size()  L, B        source_extend_vocab
            #       batch[0][4].size()  oovs, B     source的最大oovs数目 zeros_expand
            #       batch[0][5]         list(), len=B, article_oovs
            #       batch[0][6]         L, B        coverage
            #       batch[0][7]         L, B        src_sentence_flag_vec

            #   type(batch[1]): tuple len: 2
            #       batch[1][0].size()  decL, B        target
            #       batch[1][1].size()  decL, B        target_extend_vocab
            model.zero_grad()
            g_outputs, g_p_gens, g_attn, coverage_losses = model(batch)  # (decL-1, B, H)  舍弃掉最后一个得到预测输出
            # g_outputs (decL-1, B, H)
            # g_p_gens (decL-1, B, 1)
            # g_attn (decL-1, B, sourceL)
            # coverage_losses (decL-1, B)
            if opt.pointer_gen:
                targets = batch[1][1][1:]  # (decL-1, B)  舍弃掉第一个，与上面的一一对应
            else:
                targets = batch[1][0][1:]  # (decL-1, B)  舍弃掉第一个，与上面的一一对应

            enc_batch_extend_vocab = batch[0][2]
            extra_zeros = batch[0][3]
            loss, res_loss, _ = compute_loss(g_outputs, targets, model.generator, criterion, g_p_gens, g_attn,
                                              enc_batch_extend_vocab, extra_zeros, coverage_losses)
            loss.backward()
            optim.step()
            if opt.is_coverage:
                res_loss, nll_loss, coverage_report_loss = res_loss
                total_nll_loss += nll_loss
                total_coverage_report_loss += coverage_report_loss
            # update the parameters
            num_words = targets.data.ne(Constants.PAD).sum()
            report_loss += res_loss
            total_loss += res_loss
            total_words += num_words
            if i % opt.log_interval == -1 % opt.log_interval:
                if opt.is_coverage:
                    logger.info(
                        "Epoch %2d, %6d/%5d/%5d; loss: %6.2f; nll_loss: %6.2f; coverage_loss: %6.2f; %6.0f s elapsed" %
                        (epoch, totalBatchCount, i + 1, len(trainData),
                         report_loss,
                         total_nll_loss,
                         total_coverage_report_loss,
                         time.time() - start))
                    report_loss = 0
                    total_nll_loss = 0
                    total_coverage_report_loss = 0
                else:
                    logger.info(
                        "Epoch %2d, %6d/%5d/%5d; loss: %6.2f; Learning Rate: %g %g; %6.0f s elapsed" %
                        (epoch, totalBatchCount, i + 1, len(trainData),
                         report_loss,
                         optim.optimizer.param_groups[0]['lr'],
                         optim.optimizer.param_groups[1]['lr'],
                         time.time() - start))
                    report_loss = 0
                start = time.time()

            if validData is not None and totalBatchCount % opt.eval_per_batch == 0 \
                    and totalBatchCount >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                rouge_1, rouge_2 = evalModel(translator, validData)
                model.train()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                model.decoder.attn.mask = None
                logger.info('Validation Score: rouge_1 %g, rouge_2 %g' % (rouge_1 * 100, rouge_2 * 100))
                assert optim.decay_indicator == 2
                if opt.is_save:
                    if rouge_2 >= optim.best_metric[optim.decay_indicator - 1]:
                        save_model(model, optim, vocab_dicts, epoch, metric=[rouge_1, rouge_2])
                optim.updateLearningRate([rouge_1, rouge_2], epoch)

        return total_loss / float(total_words)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        logger.info('Train loss: %g   Learning Rate: %g %g' % (train_loss,
                                                               optim.optimizer.param_groups[0]['lr'],
                                                               optim.optimizer.param_groups[1]['lr']))
        logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
        # if epoch >= opt.start_decay_at and (epoch-opt.start_decay_at) % opt.decay_interval == 0:
        #     optim.updateLearningRate([0, 0], epoch)
        if opt.is_save:
            save_model(model, optim, vocab_dicts, epoch, metric=None)

def showAttention(path, s, c, attentions, index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticklabels([''] + s, rotation=90)
    ax.set_yticklabels([''] + c)
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    plt.savefig(path + str(index) + '.jpg')

def main():
    trainData, vocab_dicts = load_train_data()
    model, optim = bulid_model(vocab_dicts)
    translator = Translator(opt, model, vocab_dicts)
    validData = None
    # if opt.dev_input_src and opt.dev_ref:
    #     validData = load_dev_data(translator, opt.dev_input_src, opt.dev_ref,) # 已经分好batch Dataset， list((src_batch, tgt_batch)))
    trainModel(model, translator, trainData, validData, vocab_dicts, optim)

if __name__ == "__main__":
    main()
