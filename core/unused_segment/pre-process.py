import jieba
import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np



with open('speech.txt', 'r',encoding="UTF-8") as f:
    lines = f.readlines()
    articles = []
    titles = []
    for line in lines:
        if "'article'" in line:
            line = re.findall(r'\'article\':\s\'(.*)\'\,',line)
            a = str(line).strip('[').strip(']')
            if a not in articles:
                articles.append(a)
        if "'title'" in line:
            title = re.findall(r'\'title\':\s\'(.*)\'\,',line)
            t = str(title).strip('[').strip(']')
            if t not in titles:
                titles.append(t)


#cutting and extraction
#stopWord.txt

#重复筛选



def wordslist():
    # stop_word = [unicode(line.rstrip()) for line in open('chinese_stopword.txt')]

    for article in articles:
       seg_list = jieba.cut(article)
       result = ' '.join(seg_list)
       yield result


if __name__ == "__main__":

    wordslist = list(wordslist())

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(wordslist))

    words = vectorizer.get_feature_names()  # 所有文本的关键字
    weight = tfidf.toarray()

    print('ssss')
    n = 15 # 前五位

    # with open('TF-ITF.txt', 'w') as f:
    #     for (title, w) in zip(list(titles), weight):
    #         f.write(u'{};'.format(title))
    #         # 排序
    #         loc = np.argsort(-w)
    #         for i in range(n):
    #             f.write(u'{} {};'.format(words[loc[i]], w[loc[i]]))
    #         f.write('\n')
    # f.close()

    Tf_itf = []
    for (title, w) in zip(list(titles), weight):
        Tf_itf.append(u'{};'.format(title))
        # 排序
        loc = np.argsort(-w)
        for i in range(n):
            Tf_itf.append(u'{} {}'.format(words[loc[i]], w[loc[i]]))
        Tf_itf.append('\n')
    print(Tf_itf)

