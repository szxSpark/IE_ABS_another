import os, sys
in_file = "~/work/toutiao_word/models/seass1/dev.out.{}.txt".format(sys.argv[1])
out_file = "~/work/toutiao_word/models/seass1/detoken.dev.out.{}.txt".format(sys.argv[1])
cmd = "sed -r 's/(@@ )|(@@ ?$)//g' {} > {}".format(in_file, out_file)
os.system(cmd)

gold_file = "~/work/toutiao_word/dev/valid.title.txt"
os.system("python compute.py {} {}".format(gold_file, out_file))