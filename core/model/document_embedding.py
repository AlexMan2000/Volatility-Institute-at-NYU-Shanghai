from core.utils.text_preprocessing import TextPreprocessing
from core.model.doc2vec import Document2Vector
import os
import pandas as pd
import numpy as np
from snownlp import SnowNLP
import jieba
from gensim.models.word2vec import KeyedVectors
from simhash import Simhash
from sklearn.decomposition import PCA
import gensim
import re
from collections import Counter

from typing import List


class DocumentEmbedding:

    DEFAULT_WORD_EMBEDDING_MATRIX_PATH = "E:\VINS Project\Data\Pretrained_Embedding\embedding_matrix.npy"
    DEFAULT_WORD_EMBEDDING_MODEL_PATH = "E:\VINS Project\Data\Pretrained_Embedding\embedding_dictionary.w2v"
    DEFAULT_DOCUMENT_EMBEDDING_PATH = "E:\VINS Project\Data\Pretrained_Embedded_Document\embedding_document_matrix.npy"

    def __init__(self,train_word_embedding=False,train_document_embedding=False,re_training=False):
        self.train_document_embedding = train_document_embedding
        self.re_training = re_training
        self._load_word_embedding()
        self._extract_doc_list()

        self.embedding_dim = self.preprocessed_text.embedding_dim
        self.doc2vec = Document2Vector()

    # 加载word2vec Embedding Matrix and Embedding Model
    def _load_word_embedding(self):
        if not os.path.exists(self.DEFAULT_WORD_EMBEDDING_MODEL_PATH) or self.re_training==True:
            print("重新训练word_embedding")
            self.preprocessed_text = TextPreprocessing(training_embedding=True,training_token=True)
            self.embedding_matrix = self.preprocessed_text.get_embedding_matrix() #Embedding Matrix
            self.embedding = self.preprocessed_text.get_embedding_model() #Word2Vec
        else:
            print("加载已有数据")
            self.embedding_matrix = pd.read_csv(self.DEFAULT_WORD_EMBEDDING_MATRIX_PATH) #embedding_matrix,npy
            self.embedding = gensim.models.word2vec.Word2Vec.load(
                self.DEFAULT_WORD_EMBEDDING_MODEL_PATH).wv #Word2Vector

        # todo: get a proper word frequency for a word in a document set
        # or perhaps just a typical frequency for a word from Google's n-grams

    # 获取单词词频
    def _get_word_frequency(self, word_text=None, word=None, use_google=True):
        if use_google:
            return 0.0001  # set to a low occurring frequency - probably not unrealistic for most words, improves vector values
        else:
            assert word_text != None
            assert type(word_text) == List[str]
            self.preprocessed_text.get_word_frequency(word, word_text)

    # 将document以句子为单位切分，有两种模式
    def _extract_sentence(self, doc, auto=True, max_summary=20) -> List[List[str]]:
        """
        Return the sentence list of an article
        :param doc: a text string
        :param auto: whether to use snowNLP, set True to use SnowNLP
        :param max_summary: maximum text segments for the article's summary
        :return:
        """
        result_sentence_list=[]

        # 使用SnowNLP
        if auto:
            s = SnowNLP(doc)
            summary_list = s.summary(max_summary)
            result_sentence_list = [list(jieba.cut(i)) for i in summary_list]


        # 使用re在句号处分割句子
        else:
            sentence_list = re.findall("(.*?)。", doc)
            result_sentence_list = [list(jieba.cut(i)) for i in sentence_list]

        return result_sentence_list

    # 平滑操作, similar method proposed by Arora et al, 2016; a very strong baseline model for document embedding
    def _smoothing_method(self, sentence_list: List[List[str]], embedding_size=200, a: float = 1e-3):
        sentence_set = []
        for sentence in sentence_list:  # sentence 是List[str]
            vs = np.zeros(
                embedding_size)  # add all word2vec values into one vector for the sentence
            sentence_length = len(sentence)
            for word in sentence:
                a_value = a / (a + self._get_word_frequency(word))  # smooth inverse frequency, SIF
                try:
                    word_embedding = self.preprocessed_text.embedding[word]
                except:
                    word_embedding = self.not_found_word
                vs = np.add(vs, np.multiply(a_value,word_embedding))  # vs += sif * word_vector

            vs = np.divide(vs, sentence_length)  # weighted average
            sentence_set.append(vs)  # add to our existing re-calculated set of sentences

        # calculate PCA of this sentence set
        pca = PCA()
        pca.fit(np.array(sentence_set))
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT

        # pad the vector?  (occurs if we have less sentences than embeddings_size)
        if len(u) < embedding_size:
            for i in range(embedding_size - len(u)):
                u = np.append(u, 0)  # add needed extension for multiplication below

        # resulting sentence vectors, vs = vs -u x uT x vs
        sentence_vecs = []
        for vs in sentence_set:
            sub = np.multiply(u, vs)
            sentence_vecs.append(np.subtract(vs, sub))

        return sentence_vecs

    # Could be more complicated combined with stopword and other techniques
    def _cut_document(self,doc):
        """
        :param doc: doc是一个初始字符串
        :return: [word,word,word]
        """
        # jieba.cut here is not very convenient to use for a single document
        tmp_df = pd.DataFrame([doc],columns=["origin"])

        def jieba_cut(x):
            x = jieba.cut(x)
            return ' '.join(x)

        tmp_df["cutted"] = tmp_df["origin"].apply(jieba_cut)


        return tmp_df["cutted"][0].split(" ") # Since we are going to test it in the training set

    #获取所有文本语料
    def _extract_doc_list(self):

        self.cutted_doc_info_list = []
        doc_info_list = self.preprocessed_text.extract_dictionary()
        for sub_dictionary in doc_info_list:
            article = sub_dictionary["article"]
            title = sub_dictionary["title"]
            cutted_article = ' '.join(self._cut_document(article))
            cutted_title = ' '.join(self._cut_document(title))
            doc_info_list["article"] = cutted_article
            doc_info_list["title"] = cutted_title

        self.cutted_doc_info_list = doc_info_list

    # 转化模型
    def doc2vec_method(self,key):
        self.doc2vec.build_vocab(self.cutted_doc_info_list)
        self.doc2vec.train_model()
        doc_vec = self.doc2vec.model.docvecs[key]
        return doc_vec


    ##########################################################################################

    # 加载document_embedding(deprecated)
    def load_document_embedding(self,type):
        if not os.path.exists(self.DEFAULT_DOCUMENT_EMBEDDING_PATH) or self.train_document_embedding==True:
            print("重新训练文档Embedding")
        else:
            print("加载已有文档Embedding")

    # 获取document_embedding, unsupervised(主方法)
    def embedding_document(self,doc,key,with_tf_idf=False,with_smoothing = False,with_average=False,with_paragraph=False,a:float=1e-3):
        # Prepare the embedding for the doc using averaging, 句子的embedding dim 和word_embedding的维数一样
        document_embedded = np.zeros((1, self.embedding_dim))

        # If the word never exists in our dictionary, we randomly initialize it
        self.not_found_word = np.random.rand(1, self.embedding_dim)

        # 使用带权重的word_Embedding
        if with_tf_idf:

            cutted_doc = self._cut_document(doc)
            # 获取word_weight
            tags = self.preprocessed_text.jiebaAnalysis.extract_tags(cutted_doc, topK=None,
                                                                     withWeight=True)
            word_weight_dict = {}
            for word_cell, weight in tags:
                word_weight_dict[word_cell] = weight
            # 暂且设置topK = 150, 默认一篇文章的word 个数在150左右
            for word in doc.split(" "):
                try:
                    word_embedding = self.preprocessed_text.embedding[word]
                    word_weight = word_weight_dict[word]
                    document_embedded = np.add(document_embedded,np.multiply(word_embedding,word_weight))
                except:
                    word_embedding = self.not_found_word
                    word_weight = 0

        # sentence_level_sif_averaging, 使用sif和word_embedding对整个文章进行Embedding操作,
        if with_smoothing:
            sentence_word_list = self._extract_sentence(doc,auto=False)
            sentence_vecs = self._smoothing_method(sentence_word_list,embedding_size=self.embedding_dim)
            for sentence_vec in sentence_vecs:
                document_embedded = np.add(document_embedded,sentence_vec)
            document_embedded = np.divide(document_embedded,len(sentence_vecs))

        # word_level_average
        if with_average:
            # Cut the given document
            cutted_doc = self._cut_document(doc)
            for word in cutted_doc:
                try:
                    word_embedding = self.preprocessed_text.embedding[word]
                except:
                    word_embedding = self.not_found_word
                document_embedded=  np.add(document_embedded,word_embedding)
            document_embedded = np.divide(document_embedded,len(cutted_doc))
            return document_embedded

        # doc2vec algorithm(未完待续)
        if with_paragraph:

            return self.doc2vec_method(key)



    # 将输入格式化
    def process_embedding_for_LSTM(self,embedded):
        # 将向量pad一下
        pass


    #############################################################################################