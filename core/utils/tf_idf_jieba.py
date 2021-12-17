import math
from snownlp import SnowNLP

def extract_idf(data_content):
    idf_dic = {}
    # data_content是带分析文本，一个demo：如下图
    doc_count = len(data_content)  # 总共有多少篇文章

    for i in range(len(data_content)):
        new_content = data_content[i].split(' ')
        for word in set(new_content):
            if len(word) > 1:
                idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
            # 此时idf_dic的v值：有多少篇文档有这个词，就是多少
    for k, v in idf_dic.items():
        w = k
        p = '%.10f' % (math.log(doc_count / (1.0 + v)))  # 结合上面的tf-idf算法公式
        if w > u'\u4e00' and w <= u'\u9fa5':  # 判断key值全是中文
            idf_dic[w] = p

    with open('E:\VINS Project\Data\idf_dic.txt', 'w', encoding='utf-8') as f:
        for k in idf_dic:
            if k != '\n':

                f.write(k + ' ' + str(idf_dic[k]) + '\n')  # 写入txt文件，注意utf-8，否则jieba不认


def extract_tf(data_content):
    s = SnowNLP([i.split(" ") for i in data_content])
    return s.tf #List of dictionary, each representing the word frequency of an article.

