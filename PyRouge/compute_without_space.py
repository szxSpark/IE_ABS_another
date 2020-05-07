import os, sys
pred_file = sys.argv[1]
for line in open(pred_file, "r", encoding="utf-8"):
    print(" ".join(list("".join(line.strip().split(" ")))))
