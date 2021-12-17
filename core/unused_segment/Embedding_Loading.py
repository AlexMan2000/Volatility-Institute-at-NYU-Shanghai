import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
from gensim.models.word2vec import KeyedVectors
import numpy as np

#Too Slow if loading the whole word space
#5-10 minutes expected
DATAPATH = "./Data/Tencent_AILab_ChineseEmbedding.txt"
wv_from_text = KeyedVectors.load_word2vec_format(DATAPATH,binary=False,limit=500000)
print(wv_from_text)


#Less than 5 minutes expected
def judge_topic_from_article(tf_idf_id, k):
    industry_list = ["采掘", "化工", "钢铁", "有色金属", '建筑材料', '建筑装饰', '电气设备', '机械设备', '国防军工', '汽车', '家用电器', '轻工制造', '农林牧渔',
                     '食品饮料', '纺织服装', '医药生物', '商业贸易', '休闲服务', '电子', '计算机', '传媒', '通信', '公用事业', '交通运输', '房地产', '银行',
                     '非银金融', '综合']

    with open('./Data/TF-ITF.txt', 'r', encoding="gbk") as inF:
        all_tf_idf = inF.readlines()
        line = all_tf_idf[tf_idf_id]
        article_title = line.split(";")[0]
        word2tfidf = line.split(";")[:-1]
        industry_score_mapping = {}
        for industry in industry_list:
            similarity_list = []
            for mapping in word2tfidf:
                word = mapping.split(" ")[0]
                if word in wv_from_text:
                    similarity_list.append(wv_from_text.similarity(word, industry))
            score = np.mean(similarity_list)
            industry_score_mapping[industry] = score

    top_k = dict(sorted(industry_score_mapping.items(), key=lambda x: x[1], reverse=True)[:k])

    return industry_score_mapping, top_k

