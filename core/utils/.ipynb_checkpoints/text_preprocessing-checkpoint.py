from keras.preprocessing import text,sequence
import numpy as np
from snownlp import SnowNLP
import jieba
import jieba.analyse
import re
import pandas as pd
import os
import time
from tqdm import tqdm
import joblib
from gensim.models.word2vec import KeyedVectors
import gensim
import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
from core.utils.tf_idf_jieba import extract_idf,extract_tf


class TextPreprocessing:
    DEFAULT_SPEECH_PATH = "E:\VINS Project\Data\speech.txt"
    DEFAULT_TRAINING_DATA = "E:\VINS Project\Data\political_speech.csv"
    DEFAULT_TOKENIZER_PATH = "E:\VINS Project\Data\Pretrained_Tokenizer\Tokenizer.pkl"
    DEFAULT_EMBEDDING_DICTIONARY_PATH = "E:\VINS Project\Data\Pretrained_Embedding\embedding_dictionary.w2v"
    DEFAULT_EMBEDDING_MATRIX_PATH = "E:\VINS Project\Data\Pretrained_Embedding\embedding_matrix.npy"

    #初始化预处理器
    def __init__(self,training_token=False,training_embedding=False,embedding_dim = 200,training_tf_idf = False,use_tf_idf = True):
        self.training_token = training_token
        self.training_embedding = training_embedding
        self.embedding_dim = embedding_dim
        self.use_tf_idf = use_tf_idf
        self.training_tf_idf = training_tf_idf


        self.tokenizer = text.Tokenizer(num_words=None,
                                        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
                                        )
        #语料信息
        self.speech = open(self.DEFAULT_SPEECH_PATH, "r", encoding="utf-8")
        print("获取train_data中")
        self._load_training_data()
        print("获取train_tokenizer中")
        self._train_tokenizer()
        print("获取embedding_matrix中")
        self._train_embedding_matrix()

        self.jiebaAnalysis = jieba.analyse

        if training_tf_idf:
            print("计算tf_idf中")
            self._extract_tf_idf()
        else:
            self.jiebaAnalysis.set_idf_path("E:\VINS Project\Data\idf_dic.txt")

    #加载political_speech数据集
    def _load_training_data(self):
        if not os.path.exists(self.DEFAULT_TRAINING_DATA) or self.training_token==True:
            print("正在进行分词")
            articles = self._extract_article()
            self.training_data = pd.DataFrame({"articles": articles})
            self.training_data["cutted_articles"] = self.training_data["articles"].apply(lambda i: jieba.cut(i))
            self.training_data["cutted_articles"] = [' '.join(i) for i in self.training_data["cutted_articles"]]
        else:
            print("加载已有数据")
            self.training_data = pd.read_csv(self.DEFAULT_TRAINING_DATA)


    def _load_stopwords(self):
        with open("./Data/stopwords-master/复旦_stopwords.txt", encoding="utf-8") as tmpfile:
            stopwords = tmpfile.readlines()
            tmpfile.close()

        return [word.strip() for word in stopwords]


    def _jieba_clean_cut(self):
        stopwords = self._load_stopwords()
        unsigned_text = re.sub("[\s+\.\!\/_,$%^*“”(+\"\']+|[+——！，。？、~@#￥%……&*（）：《》0-9]+", "", text)
        seg_list = jieba.cut(unsigned_text, cut_all=False)
        seg_list = list(seg_list)
        seg_list = list(set(seg_list) - set(stopwords))
        return seg_list

    #获取所有的article
    def _extract_article(self):
        '''
        :param file:
        :return: 所有文章
        '''
        file = self.speech
        lines = file.readlines()
        articles = set()
        for line in lines:
            if "'article'" in line:
                line = re.findall(r'\'article\':\s\'(.*)\'\,', line)
                if len(line) > 0:
                    if line[0] != '':
                        articles.add(line[0])

        return list(articles)

    #对数据集进行文本分词训练用于LSTM注入
    def _train_tokenizer(self):
        if not os.path.exists(self.DEFAULT_TOKENIZER_PATH) or self.training_token:
            print("正在训练Tokenizer")
            articles = list(self.training_data["cutted_articles"])
            self.tokenizer.fit_on_texts(articles)
            joblib.dump(self.tokenizer,self.DEFAULT_TOKENIZER_PATH)
        else:
            print("加载预训练Tokenizer")
            self.tokenizer = joblib.load(self.DEFAULT_TOKENIZER_PATH)

    #获取训练Embedding(未测试)
    def _train_embedding_matrix(self):
        if not os.path.exists(self.DEFAULT_EMBEDDING_DICTIONARY_PATH) or self.training_embedding:
            print("正在训练Embedding Matrix, could take a while")
            X = self.training_data["cutted_articles"]
            X = [i.split() for i in X] # Take a while
            model = gensim.models.Word2Vec(X,min_count=5,
                                                    window=8,
                                                    vector_size=200)
            model.save(self.DEFAULT_EMBEDDING_DICTIONARY_PATH)
            self.embedding = model.wv
            word_index = self.tokenizer.word_index

            print("这是word_index",len(word_index))
            print("开始创建Embedding Matrix")
            self.embedding_matrix = np.zeros((len(word_index)+1,self.embedding_dim))
            embedding_dict = dict(zip(self.embedding.index_to_key, self.embedding.vectors))
            for word, i in tqdm(word_index.items()):
                embedding_vector = embedding_dict.get(word, None)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector

            print("创建完毕Embedding Matrix")

            joblib.dump( self.embedding_matrix,self.DEFAULT_EMBEDDING_MATRIX_PATH)

        else:
            print("加载预训练Embedding Matrix")
            self.embedding = gensim.models.word2vec.Word2Vec.load(
                self.DEFAULT_EMBEDDING_DICTIONARY_PATH).wv
            self.embedding_matrix = joblib.load(self.DEFAULT_EMBEDDING_MATRIX_PATH)


    def _extract_tf_idf(self):
        extract_idf(list(self.training_data["cutted_articles"]))
        self.jiebaAnalysis.set_idf_path("E:\VINS Project\Data\idf_dic.txt")

    ##############################################################################################
    def get_tokenizer(self):
        return self.tokenizer

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def get_embedding_model(self):
        return self.embedding

    def get_word_frequency(self,word,text):
        """
        :param text: Should be a string of cutted words
        :return:
        """
        s = SnowNLP([text.split(" ")])
        return s.tf[0][word]

    def tokenize_for_text(self,text=None,maxlen = 20):
        """
        Transform a cutted string of text into padded seqence for LSTM input
        :param text: Could be a single string, however, list of cutted strings are recommended
        :return: tokened_text, np.ndarray, [text_id,maxlen]
        """
        if not text:
            text_seq = self.tokenizer.texts_to_sequences(list(self.training_data["cutted_articles"]))
            text_pad = sequence.pad_sequences(text_seq,maxlen=maxlen)
        else:
            # If you only want to tokenize one sentence
            if type(text)==str:
                text = [text]

            text_seq = self.tokenizer.texts_to_sequences(text)  # [tokenized words]
            text_pad = sequence.pad_sequences(text_seq, maxlen=maxlen) #默认在前面补零凑满max_len长度

        return text_pad

    #未测试
    def get_tag_weight_for_text(self,text_id=None,topK = 20,withWeight=True):
        """
        :param text_id: 文章的序号
        :param topK: 返回topK个最重要的分词
        :param withWeight: 带有权重
        :return: [{topword1:weight,topword2:weight,.....,}]
        """
        if not text_id:
            result_list = []
            #返回所有文档的topK_weight的词语,用于后续使用LSTM文章Embedding
            for cutted_text in list(self.training_data["cutted_articles"]):
                tags = self.jiebaAnalysis.extract_tags(cutted_text,topK=topK,withWeight=withWeight)
                tmp_dict={}
                for word,weight in tags:
                    tmp_dict[word] = weight
                print(tmp_dict)
                result_list.append(tmp_dict)
        else:
            result_list=[]
            tags = self.jiebaAnalysis.extract_tags(list(self.training_data["cutted_articles"])[text_id],topK=topK,withWeight=withWeight)
            tmp_dict={}
            for word,weight in tags:
                tmp_dict[word] = weight
            result_list.append(tmp_dict)
        return result_list

    ##############################################################################################
    def test_tokenized(self):
        print(self.tokenizer.word_counts)
        print(self.tokenizer.word_index)

    def test_padded(self):
        print(self.tokenize_for_text("世界级 国家博物馆"))

    def test_embedding(self):
        print(self.embedding_matrix)



if __name__=="__main__":
    textCutter = TextPreprocessing(training_embedding=False,training_tf_idf=False)
    # print(textCutter.get_tag_weight_for_text(6,20,True))
    # # print(textCutter.get_tag_weight_for_text())
    print(textCutter.tokenize_for_text())