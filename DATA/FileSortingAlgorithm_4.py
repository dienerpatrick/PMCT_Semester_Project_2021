from shutil import copy2

LValidTest = "Sorted_Data\L_Valid\L_Test\L_Test"
LValidTrain = "Sorted_Data\L_Valid\L_Train\L_Train"
APValidTest = "Sorted_Data\AP_Valid\AP_Test\AP_Test"
APValidTrain = "Sorted_Data\AP_Valid\AP_Train\AP_Train"

LSortedTest = "Sorted_Data\L_Sorted_4\L_Test"
LSortedTrain = "Sorted_Data\L_Sorted_4\L_Train"
APSortedTest = "Sorted_Data\AP_Sorted_4\AP_Test"
APSortedTrain = "Sorted_Data\AP_Sorted_4\AP_Train"

with open("labels.txt") as labels:
    for line in labels.readlines()[1:]:
        pair = line.strip("\n").split("\t")
        if len(pair) == 2:
            try:
                if pair[1] == '':
                    pass
                elif int(pair[1]) < 25:
                    try:
                        copy2(APValidTrain + "/APminipSort_" + pair[0] + ".png",
                              APSortedTrain + "\AP_0_25" + "\APSort4_" + pair[0] + ".png")
                        copy2(LValidTrain + "/LminipSort_" + pair[0] + ".png",
                              LSortedTrain + "\L_0_25" + "\LSort4_" + pair[0] + ".png")
                    except FileNotFoundError:
                        copy2(APValidTest + "/APminipSort_" + pair[0] + ".png",
                              APSortedTest + "\AP_0_25" + "\APSort4_" + pair[0] + ".png")
                        copy2(LValidTest + "/LminipSort_" + pair[0] + ".png",
                              LSortedTest + "\L_0_25" + "\LSort4_" + pair[0] + ".png")

                elif 25 <= int(pair[1]) < 50:
                    try:
                        copy2(APValidTrain + "/APminipSort_" + pair[0] + ".png",
                              APSortedTrain + "\AP_25_50" + "\APSort4_" + pair[0] + ".png")
                        copy2(LValidTrain + "/LminipSort_" + pair[0] + ".png",
                              LSortedTrain + "\L_25_50" + "\LSort4_" + pair[0] + ".png")
                    except FileNotFoundError:
                        copy2(APValidTest + "/APminipSort_" + pair[0] + ".png",
                              APSortedTest + "\AP_25_50" + "\APSort4_" + pair[0] + ".png")
                        copy2(LValidTest + "/LminipSort_" + pair[0] + ".png",
                              LSortedTest + "\L_25_50" + "\LSort4_" + pair[0] + ".png")

                elif 50 <= int(pair[1]) < 75:
                    try:
                        copy2(APValidTrain + "/APminipSort_" + pair[0] + ".png",
                              APSortedTrain + "\AP_50_75" + "\APSort4_" + pair[0] + ".png")
                        copy2(LValidTrain + "/LminipSort_" + pair[0] + ".png",
                              LSortedTrain + "\L_50_75" + "\LSort4_" + pair[0] + ".png")
                    except FileNotFoundError:
                        copy2(APValidTest + "/APminipSort_" + pair[0] + ".png",
                              APSortedTest + "\AP_50_75" + "\APSort4_" + pair[0] + ".png")
                        copy2(LValidTest + "/LminipSort_" + pair[0] + ".png",
                              LSortedTest + "\L_50_75" + "\LSort4_" + pair[0] + ".png")

                elif 75 <= int(pair[1]) <= 100:
                    try:
                        copy2(APValidTrain + "/APminipSort_" + pair[0] + ".png",
                              APSortedTrain + "\AP_75_100" + "\APSort4_" + pair[0] + ".png")
                        copy2(LValidTrain + "/LminipSort_" + pair[0] + ".png",
                              LSortedTrain + "\L_75_100" + "\LSort4_" + pair[0] + ".png")
                    except FileNotFoundError:
                        copy2(APValidTest + "/APminipSort_" + pair[0] + ".png",
                              APSortedTest + "\AP_75_100" + "\APSort4_" + pair[0] + ".png")
                        copy2(LValidTest + "/LminipSort_" + pair[0] + ".png",
                              LSortedTest + "\L_75_100" + "\LSort4_" + pair[0] + ".png")

            except FileNotFoundError:
                print(f"file {pair[0]} not found")