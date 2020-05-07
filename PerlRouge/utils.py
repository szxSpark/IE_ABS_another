from pyrouge import Rouge155
import os, shutil, string
import numpy as np
import sys
import tempfile
tempfile.tempdir = "./temp"
summaries = [
    ['北 京 精 神 病 人 被 用 束 缚 带 绑 床 上 ， 遭 病 友 掐 死 ； 法 院 认 定 医 院 管 理 不 当 、 值 班 护 士 失 职 ， 构 成 医 疗 事 故 罪 。'],
    ['世 界 卫 生 组 织 发 布 报 告 称 ， 在 过 去 1 0 年 ， 自 杀 取 代 难 产 死 亡 ， 成 为 全 球 年 轻 女 性 死 亡 的 最 主 要 原 因'],
    ['温 州 瑞 安 重 度 烧 伤 孕 妇 已 有 清 晰 意 识 ， 急 需 A 型 血 小 板 ； 5 月 9 日 其 煮 夜 宵 时 发 生 爆 燃 ， 已 收 捐 款 超 6 0 0 万 。'],
    ['据 雅 虎 体 育 ， 莫 里 斯 - 威 廉 姆 斯 与 骑 士 达 成 协 议 ， 将 与 詹 皇 重 聚 ， 他 将 签 下 2 年 4 3 0 万 美 元 的 合 同'],
    ['天 津 ： 受 媒 体 松 绑 ， 天 津 ： 沪 深 四 区 今 起 取 消 ； 媒 体 称 “ 光 印 户 口 落 户 天 津 ” ， ']
]

references = [
    [['北 京 精 神 病 人 被 用 束 缚 带 绑 床 上 ， 遭 病 友 掐 死 ； 法 院 认 定 医 院 管 理 不 当 、 值 班 护 士 失 职 ， 构 成 医 疗 事 故 罪 。']],
    [['世 界 卫 生 组 织 发 布 报 告 称 ， 在 过 去 1 0 年 ， 自 杀 取 代 难 产 死 亡 ， 成 为 全 球 年 轻 女 性 死 亡 的 最 主 要 原 因']],
    [['温 州 瑞 安 重 度 烧 伤 孕 妇 已 有 清 晰 意 识 ， 急 需 A 型 血 小 板 ； 5 月 9 日 其 煮 夜 宵 时 发 生 爆 燃 ， 已 收 捐 款 超 6 0 0 万 。']],
    [['据 雅 虎 体 育 ， 莫 里 斯 - 威 廉 姆 斯 与 骑 士 达 成 协 议 ， 将 与 詹 皇 重 聚 ， 他 将 签 下 2 年 4 3 0 万 美 元 的 合 同']],
    [['天 津 ： 5 月 3 1 日 起 外 地 人 买 房 不 再 送 户 口 ， 取 消 蓝 印 户 口 政 策 ， 以 “ 积 分 落 户 ” 取 代 。']]
]

def load_pairs(sum_file, ref_file, is_cn):
    summaries = []
    references = []
    sum_f = open(sum_file, "r", encoding="utf-8")
    ref_f = open(ref_file, "r", encoding="utf-8")
    while True:
        sline = sum_f.readline().strip()
        tline = ref_f.readline().strip()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break
        if is_cn:
            sline = " ".join(list("".join(sline.strip().split(" "))))
            sline = sline.replace("< u n k >", "<unk>")

        summaries.append([sline])

        references.append([[tline]])
    return summaries, references

def convert_format(summaries, references):
    def build_dict():
        sentences = []
        assert len(summaries) == len(references)  # data_num
        for i, (summary, candidates) in enumerate(zip(summaries, references)):
            for j, candidate in enumerate(candidates):
                # candidate: list(str) 多个文摘句
                assert type(candidate) == list
                sentences += candidate
            assert type(summary) == list
            sentences += summary
        vocab = {}
        for sentence in sentences:
            words = sentence.split(" ")
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    vocab = build_dict()
    summaries_idx = []
    references_idx = []
    for i, (summary, candidates) in enumerate(zip(summaries, references)):
        ref_idx_tmp = []
        for j, candidate in enumerate(candidates):
            # candidate: list(str) 多个文摘句
            assert type(candidate) == list
            candidate = list(map(lambda x:" ".join([str(vocab[word]) for word in x.split(" ")]),
                                 candidate))
            ref_idx_tmp.append(candidate)
        references_idx.append(ref_idx_tmp)
        assert type(summary) == list
        sum_idx_tmp = list(map(lambda x: " ".join([str(vocab[word]) for word in x.split(" ")]),
                               summary))
        summaries_idx.append(sum_idx_tmp)
    return summaries_idx, references_idx


def evaluate_rouge(summaries, references, remove_temp=True, rouge_args=[]):
    '''
    Args:
        summaries: [[sentence]]. Each summary is a list of strings (sentences)
        references: [[[sentence]]]. Each reference is a list of candidate summaries.
        remove_temp: bool. Whether to remove the temporary files created during evaluation.
        rouge_args: [string]. A list of arguments to pass to the ROUGE CLI.
    '''
    # summaries: data_num, 3, str
    temp_chars = string.ascii_uppercase + string.digits
    temo_chars_idx = np.random.choice(a=len(temp_chars), size=10, replace=True, p=None)
    temp_dir = ''.join([temp_chars[idx] for idx in temo_chars_idx])
    temp_dir = os.path.join("temp", temp_dir)
    print(temp_dir)
    system_dir = os.path.join(temp_dir, 'system')
    model_dir = os.path.join(temp_dir, 'model')
    # directory for generated summaries
    os.makedirs(system_dir)
    # directory for reference summaries
    os.makedirs(model_dir)
    print(temp_dir, system_dir, model_dir)

    assert len(summaries) == len(references)  # data_num
    for i, (summary, candidates) in enumerate(zip(summaries, references)):
        summary_fn = '%i.txt' % i
        for j, candidate in enumerate(candidates):
             # candidate: list(str) 多个文摘句
            candidate_fn = '%i.%i.txt' % (i, j)
            with open(os.path.join(model_dir, candidate_fn), 'w', encoding="utf-8") as f:
                # print(candidate), 参考的abstract
                f.write('\n'.join(candidate))

        with open(os.path.join(system_dir, summary_fn), 'w', encoding="utf-8") as f:
            # 模型生成的
            f.write('\n'.join(summary))  # 生成的3个文摘的句子

    args_str = ' '.join(map(str, rouge_args))
    rouge = Rouge155(rouge_args=args_str)  # 怎么用
    rouge.system_dir = system_dir
    rouge.model_dir = model_dir
    rouge.system_filename_pattern = '(\d+).txt'  # 可以识别system_dir
    rouge.model_filename_pattern = '#ID#.\d+.txt'  # '#ID#' 用于对齐文章

    #rouge_args = '-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a'
    #output = rouge.convert_and_evaluate(rouge_args=rouge_args)
    output = rouge.convert_and_evaluate()
    r = rouge.output_to_dict(output)  # dict

    # remove the created temporary files
    if remove_temp:
       shutil.rmtree(temp_dir)
    return r

def compute_rouge(summaries, references):
    summaries_idx, references_idx = convert_format(summaries, references)
    r = evaluate_rouge(summaries_idx, references_idx, remove_temp=True, rouge_args=[])
    return {
        'rouge-1': r['rouge_1_f_score'],
        'rouge-2': r['rouge_2_f_score'],
        'rouge-l': r['rouge_l_f_score']
    }

if __name__ == "__main__":
    dir_name, ref_file = sys.argv[1], sys.argv[2]
    # ref_file: [toutiao.valid.title.txt, lcsts.valid.title.txt, giga.valid.title.txt]
    if "giga" in ref_file:
        is_cn = False
    else:
        is_cn = True
    sum_file_list = []
    for root, _, files in os.walk(dir_name):
        for name in files:
            if 'log' in name:
                continue
            sum_file_list.append(os.path.join(root, name))

    result = []
    for sum_file in sum_file_list:
        summaries, references = load_pairs(sum_file=sum_file, ref_file=ref_file, is_cn=is_cn)
        if is_cn:
            summaries_idx, references_idx = convert_format(summaries, references)
            print(sum_file.split(os.sep)[-1], len(summaries_idx), len(references_idx))
            r = evaluate_rouge(summaries_idx, references_idx, remove_temp=True, rouge_args=[])
        else:
            r = evaluate_rouge(summaries, references, remove_temp=True, rouge_args=[])
        result.append((sum_file,  r['rouge_1_f_score'], r['rouge_2_f_score'],  r['rouge_l_f_score']))

    result = sorted(result, key=lambda x:int(x[0].split(".")[-2]), reverse=False)

    for r in result:
        print("{} rouge_1:{} rouge_2:{} rouge_l:{}".format(r[0].split(os.sep)[-1],r[1],r[2],r[3]))

