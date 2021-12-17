from scipy import spatial
def similarity(w1,w2):
    with open('./Data/Tencent_AILab_ChineseEmbedding.txt', 'r',encoding="UTF-8") as inF:
        for line in inF:
            if line.split(" ")[0] == w1:
                global l_first
                l_first = [float(i) for i in line.split(" ")[1:]]
                break
        for line in inF:
            if line.split(" ")[0] == w2:
                global l_second
                l_second = [float(i) for i in line.split(" ")[1:]]
                break
        sim = 1 - spatial.distance.cosine(l_first, l_second)
        print(f"The cosine similarity between {w1} and {w2} is {sim}.")
similarity("经济","金融")
similarity("经济","建筑装饰")
#
# industry_list = ["采掘","化工","钢铁","有色金属",'建筑材料','建筑装饰','电气设备','机械设备','国防军工','汽车','家用电器','轻工制造','农林牧渔','食品饮料','纺织服装','医药生物','商业贸易','休闲服务','电子','计算机','传媒','通信','公用事业','交通运输','房地产','银行','非银金融','综合']
# industry_list = ["采掘","化工"]
# f = open("TF-ITF.txt","r",encoding="gbk")
# l11 = f.readline().split(";")
# print(l11[1].split(" ")[0])
# print(l11[1].split(" ")[1])
# f.close()
#
# with open('./Data/TF-ITF.txt', 'r', encoding="gbk") as inF:
#     for line in inF:
#         line_new = line.split(";")
#         similarity(line_new[1].split(" ")[0],"经济")

similarity("厄方","经济")
