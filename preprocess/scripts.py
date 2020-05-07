import json
from utils.Constants import E1_R_WORD, R_E2_WORD
def func1():
    # 构建subword的svo
    # hier_json = "/home/zxsong/workspace/seass/data/toutiao_word/train/train.hierarchical.combine.json"
    hier_json = "/home/zxsong/workspace/seass/data/toutiao_word/dev/valid.hierarchical.combine.json"

    id2svo = {}
    for data in json.load(open(hier_json, "r", encoding="utf-8")):
        article, abstract = data['article'], data['abstract']
        flat_abstract = []
        for abs_sent in abstract:
            flat_abstract.extend(abs_sent)
        flat_article = []
        for art_sent in article:
            flat_article.extend(art_sent)
        svo = data["svo"]
        svo_words = []  # 每个集合是一个实体列表
        for e1, r, e2 in svo:
            e1 = [" ".join(e1)]
            r = [" ".join(r)]
            e2 = [" ".join(e2)]
            svo_words.append(e1 + r + e2)
        origin_id = data['origin_id']
        id2svo[origin_id] = svo_words
    print(len(id2svo))
    # data_f = "/home/zxsong/workspace/seass/data/toutiao_word/train/train.title.txt"
    # out_f = "/home/zxsong/workspace/seass/data/toutiao_word/train/subword/svo.tmp.txt"
    data_f = "/home/zxsong/workspace/seass/data/toutiao_word/dev/valid.title.txt"
    out_f = "/home/zxsong/workspace/seass/data/toutiao_word/dev/subword/svo.tmp.txt"
    c = 0
    with open(out_f, "w", encoding="utf-8") as f:
        for i, title in enumerate(open(data_f, "r", encoding="utf-8")):
            if i in id2svo:
                for _svo in id2svo[i]:
                    c += 1
                    f.write("\n".join(_svo) + "\n")
            else:
                pass
    print(c)

def func2():
    # 逆操作
    # hier_json = "/home/zxsong/workspace/seass/data/toutiao_word/train/train.hierarchical.combine.json"
    hier_json = "/home/zxsong/workspace/seass/data/toutiao_word/dev/valid.hierarchical.combine.json"
    id2svo_num = {}
    for data in json.load(open(hier_json, "r", encoding="utf-8")):
        svo = data["svo"]
        origin_id = data['origin_id']
        id2svo_num[origin_id] = len(svo)

    # data_f = "/home/zxsong/workspace/seass/data/toutiao_word/train/train.title.txt"
    data_f = "/home/zxsong/workspace/seass/data/toutiao_word/dev/valid.title.txt"
    data_num = 0
    for i, title in enumerate(open(data_f, "r", encoding="utf-8")):
        data_num += 1

    # in_f = "/home/zxsong/workspace/seass/data/toutiao_word/train/subword/svo.tmp.subword.txt"
    in_f = "/home/zxsong/workspace/seass/data/toutiao_word/dev/subword/svo.tmp.subword.txt"

    c = 0
    one_svo = ""
    final_svo = []
    for line in open(in_f, "r", encoding="utf-8"):
        c += 1
        one_svo += line.strip()
        if c % 3 == 0:
            # 一个svo
            final_svo.append(one_svo)
            one_svo = ""
        elif c % 3 == 1:
            # s
            one_svo += " "+E1_R_WORD+" "
        else:
            # s, v
            one_svo += " "+R_E2_WORD+" "
    assert c % 3 == 0
    i = 0
    # with open("/home/zxsong/workspace/seass/data/toutiao_word/train/subword/svo.subword.txt",
    with open("/home/zxsong/workspace/seass/data/toutiao_word/dev/subword/svo.subword.txt",
              "w", encoding="utf-8")as f:
        for data_i in range(data_num):
            svo_num = id2svo_num[data_i] if data_i in id2svo_num else 0
            svos = []
            for _ in range(svo_num):
                svos.append(final_svo[i])
                i += 1
            f.write(json.dumps(svos, ensure_ascii=False)+"\n")
    assert i == len(final_svo)


if __name__ == "__main__":
    func1()
    func2()