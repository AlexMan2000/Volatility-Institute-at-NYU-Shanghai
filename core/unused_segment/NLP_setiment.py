#coding:utf-8
from snownlp import *
import re
import numpy as np

f = open("./Data/speech.txt","r",encoding="utf-8")


def extract_dictionary(file):
    dictionary_list = []
    assert file != None
    txt = file.read()
    assert txt != None
    dictionary_txt = re.findall("{'article'.*?'}",txt,re.S)
    for sub_txt in dictionary_txt:
        tmp_list = re.findall(r"\'(?P<key>.*?)\':\s['|[]?(?P<value>.*)['|]]?", sub_txt)
        tmp_dict = dict(tmp_list)
        tmp_dict["date"] = re.sub("'","",tmp_dict["date"])
        dictionary_list.append(tmp_dict)
    return dictionary_list

def extract_article(file):
    lines = file.readlines()
    articles = set()
    for line in lines:
        if "'article'" in line:
            line = re.findall(r'\'article\':\s\'(.*)\'\,', line)
            if len(line) > 0:
                if line[0]!='':
                    articles.add(line[0])

    return list(articles)


def extract_title(file):
    lines = f.readlines()
    titles = set()
    for line in lines:
        if "'title'" in line:
            line = re.findall(r'\'title\':\s\'(.*)\'\,', line)
            if len(line) > 0:
                if line[0]!='':
                    titles.add(line[0])

    return list(titles)



def sentiment_from_summary(id,f,average=5):


    article = extracted_dictionary[id]["article"]
    title = extracted_dictionary[id]["title"]
    sa = SnowNLP(article)
    key_words = sa.keywords(5)
    print(f"Key words for article with id {id}, {key_words}")
    key_extraction = sa.summary(average)
    print(f"Title for article with id {id}: {title}")
    print("===============================================")
    print(f"Summary for the article: {key_extraction}")
    score_list = []
    for key in key_extraction:
        tmp = SnowNLP(key)
        score_list.append(tmp.sentiments)

    print(f"The sentiment for this article, {np.mean(score_list)} where >0.4 positive")


extracted_dictionary = extract_dictionary(f)

if __name__ =="__main__":
    sentiment_from_summary(700,f,5)
    f.close()