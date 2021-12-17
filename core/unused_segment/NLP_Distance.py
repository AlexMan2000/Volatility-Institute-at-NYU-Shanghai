from simhash import Simhash
from NLP_setiment import extract_dictionary,extract_article,extracted_dictionary
import re
import jieba

f = open("./Data/speech.txt","r",encoding="utf-8")


def document_distance(id1,id2):
    info1 = extracted_dictionary[id1]
    info2 = extracted_dictionary[id2]

    article1 = info1["article"]
    article2 = info2["article"]

    article1 = jieba_clean_cut(article1)
    article2 = jieba_clean_cut(article2)

    hash1 = Simhash(article1)
    hash2 = Simhash(article2)

    return hash1.distance(hash2)


def load_stopwords():
    with open("./Data/stopwords-master/baidu_stopwords.txt",encoding="utf-8") as tmpfile:
        stopwords = tmpfile.readlines()
        tmpfile.close()

    return [word.strip() for word in stopwords]



def jieba_clean_cut(text):
    stopwords = load_stopwords()
    unsigned_text = re.sub("[\s+\.\!\/_,$%^*“”(+\"\']+|[+——！，。？、~@#￥%……&*（）：《》0-9]+", "",text)
    seg_list = jieba.cut(unsigned_text, cut_all=False)
    seg_list = list(seg_list)
    seg_list = list(set(seg_list)-set(stopwords))

    return seg_list



def find_k_most_similar(id,k):
    score_list = []
    for i in range(len(extracted_dictionary)):
        if i != id:
            score_list.append((i,document_distance(id,i)))

    score_list.sort(key=lambda x: x[1],reverse=True)

    print(f"The top {k} similar articles to {extracted_dictionary[id]['title']} are:\n")
    for i in range(k):
        print(extracted_dictionary[score_list[i][0]]["title"])


if __name__ == "__main__":
    # find_k_most_similar(0,5)
    # print(document_distance(5,6))
    pass