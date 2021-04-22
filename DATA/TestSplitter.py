from shutil import copy2
import os

src = "Lminip"
aptest = "Sorted_Data\AP_Sorted\AP_Test"
ltrain = "Sorted_Data\L_Sorted\L_Train"
ltest = "Sorted_Data\L_Sorted\L_Test"


for filename in os.listdir(aptest + "/AP0_50"):
    ID = filename[-8:-4]
    copy2(os.path.join(ltrain, "L0_50", "LminipSort_" + ID + ".png"),
          os.path.join(ltest, "L0_50", "LminipSort_" + ID + ".png"))
    os.remove(os.path.join(ltrain, "L0_50", "LminipSort_" + ID + ".png"))

for filename in os.listdir(aptest + "/AP50_100"):
    ID = filename[-8:-4]
    copy2(os.path.join(ltrain, "L50_100", "LminipSort_" + ID + ".png"),
          os.path.join(ltest, "L50_100", "LminipSort_" + ID + ".png"))
    os.remove(os.path.join(ltrain, "L50_100", "LminipSort_" + ID + ".png"))
