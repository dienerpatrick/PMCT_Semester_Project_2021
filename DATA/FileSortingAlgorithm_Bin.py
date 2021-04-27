from shutil import copy2

src = "Lminip"
dst1 = "Sorted_Data\L0_50"
dst2 = "Sorted_Data\L50_100"
dst3 = "Sorted_Data\LNONE"

srcpref = "/Lminip_"
dstpref = "/LminipSort_"

with open("labels.txt") as labels:
    for line in labels.readlines()[1:]:
        pair = line.strip("\n").split("\t")
        if len(pair) == 2:
            try:
                if pair[1] == '':
                    copy2(src + srcpref + pair[0] + ".png", dst3 + dstpref + pair[0] + ".png")
                elif int(pair[1]) < 50:
                    copy2(src + srcpref + pair[0] + ".png", dst1 + dstpref + pair[0] + ".png")
                elif int(pair[1]) >= 50:
                    copy2(src + srcpref + pair[0] + ".png", dst2 + dstpref + pair[0] + ".png")
            except FileNotFoundError:
                print(f"file {pair[0]} not found")