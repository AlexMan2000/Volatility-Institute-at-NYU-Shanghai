#coding: utf-8
import numpy as np
from snownlp import SnowNLP
import jieba
import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
from gensim.models.word2vec import KeyedVectors
import gensim
import re
from simhash import Simhash
import os
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize


from core.utils.LogisticUtils import NLPNormalizeVectorizer
from core.utils.loss_functions import multiclass_logloss

class NLP_Text_Analysis:

    DEFAULT_PROJECT_PATH = "E:/VINS Project"
    DEFAULT_TRAINING_DATA = "./Data/复旦大学中文文本分类语料.xlsx"
    DEFAULT_SEGMENTED_TRAINING_DATA = "./Data/文本分词.xlsx"
    DEFAULT_SPEECH_PATH = "./Data/speech.txt"
    DEFAULT_EMBEDDING_PATH = "./Data/Tencent_AILab_ChineseEmbedding.txt"
    DEFAULT_TF_IDF_PATH = "./Data/TF-ITF.txt"
    DEFAULT_STOP_WORDS_PATH = "./Data/stopwords-master/baidu_stopwords.txt"
    DEFAULT_PRETRAINED_MODEL_ROOT_PATH = "./Data/Pretrained_Model"
    DEFAULT_PRETRAINED_TRANSFORMER_ROOT_PATH = "./Data/Pretrained_Transformer"
    DEFAULT_PRETRAINED_MATRICES_ROOT_PATH = "./Data/Pretrained_Matrices"

    def __init__(self,speech=DEFAULT_SPEECH_PATH,tf_idf = DEFAULT_TF_IDF_PATH,stop_words = DEFAULT_STOP_WORDS_PATH,training_embedding=False,load_embedding=False,embedding_path=None,embedding_limit=500000,training_model=True):
        '''

        :param speech: 语料
        :param tf_idf: 词频
        :param stop_words: 停用词
        :param embedding: Tecent编码
        :param embedding_path: 编码路径
        :param embedding_limit: 编码加载的单词数量
        '''
        self.embedding = False
        if load_embedding:
            print("Loading the embedding matrix could take a while")
            if embedding_path:
                embed = embedding_path
            else:
                embed = self.DEFAULT_EMBEDDING_PATH
            self.embedding = KeyedVectors.load_word2vec_format(embed,binary=False,limit=embedding_limit)

        self.tf_idf = open(self.DEFAULT_TF_IDF_PATH,"r",encoding="utf-8")
        self.speech = open(self.DEFAULT_SPEECH_PATH,"r",encoding="utf-8")
        self.stop_words = open(self.DEFAULT_STOP_WORDS_PATH,"r",encoding="utf-8")
        if training_model:
            self._load_training_data()
            self.generate_random()
        self.extracted_dictionary = self._extract_dictionary(self.speech)
        self.training_embedding = training_embedding

    # 获取文章所有信息，以Map集合储存
    def _extract_dictionary(self,file=None):
        if file:
            dictionary_list = []
            assert file != None
            txt = file.read()
            assert txt != None
            dictionary_txt = re.findall("{'article'.*?'}", txt, re.S)
            for sub_txt in dictionary_txt:
                tmp_list = re.findall(r"\'(?P<key>.*?)\':\s['|[]?(?P<value>.*)['|]]?", sub_txt)
                tmp_dict = dict(tmp_list)
                tmp_dict["date"] = re.sub("'", "", tmp_dict["date"])
                dictionary_list.append(tmp_dict)
            return dictionary_list

    # 文档距离
    def _document_distance(self,id1, id2):
        info1 = self.extracted_dictionary[id1]
        info2 = self.extracted_dictionary[id2]

        article1 = info1["article"]
        article2 = info2["article"]

        article1 = self._jieba_clean_cut(article1)
        article2 = self._jieba_clean_cut(article2)

        hash1 = Simhash(article1)
        hash2 = Simhash(article2)

        return hash1.distance(hash2)

    # 文本分词
    def _jieba_clean_cut(self,text):
        stopwords = self._load_stopwords()
        unsigned_text = re.sub("[\s+\.\!\/_,$%^*“”(+\"\']+|[+——！，。？、~@#￥%……&*（）：《》0-9]+", "", text)
        seg_list = jieba.cut(unsigned_text, cut_all=False)
        seg_list = list(seg_list)
        seg_list = list(set(seg_list) - set(stopwords))

        return seg_list

    # 加载停用词
    def _load_stopwords(self):
        with open("./Data/stopwords-master/复旦_stopwords.txt", encoding="utf-8") as tmpfile:
            stopwords = tmpfile.readlines()
            tmpfile.close()

        return [word.strip() for word in stopwords]

    # 该函数会将语句转化为一个标准化的向量（Normalized Vector）
    def _sent2vec(self,sentence):
        stwlist = self._load_stopwords()
        # jieba.enable_parallel()  # 并行分词开启
        # print(sentence)
        # words = str(sentence).lower()
        # print(words)
        words = sentence.split(" ") # list cut
        words = [w for w in words if not w in stwlist]
        # print(words)
        # print(self.embedding)
        M = []
        for w in words:
            try:
                M.append(self.embedding[w])
            except:
                # Embedding中没有就跳过
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        # print(type(v))
        if type(v) != np.ndarray:
            print("#########################")
            return np.zeros(200)
        #返回一个200维的向量
        return v / np.sqrt((v ** 2).sum())

    # 加载训练集，初始化
    def _load_training_data(self):
        if not os.path.exists(self.DEFAULT_SEGMENTED_TRAINING_DATA):
            self.training_data = pd.read_excel(self.DEFAULT_TRAINING_DATA,"sheet1")
            #jieba.enable_parallel(64)  # 并行分词开启, WINDOWS不支持，colab可以
            self.training_data['文本分词'] = self.training_data['正文'].apply(lambda i: jieba.cut(i))
            # 转换成以字符串为单元的分词结构
            self.training_data['文本分词'] = [' '.join(i) for i in self.training_data['文本分词']]
        else:
            print("加载已有数据")
            self.training_data = pd.read_excel(self.DEFAULT_SEGMENTED_TRAINING_DATA,sheet_name="sheet1")

        # 实例化一个LabelEncoder(), 将离散的分类转换成数值类型
        self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder = self.labelEncoder.fit(self.training_data["分类"].values)
        self.labels = self.labelEncoder.transform(self.training_data.分类.values)

    # 预处理数据
    def _preprocess_training_data(self):

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.training_data.文本分词,
                                                        self.labels,
                                                        stratify=self.labels,  # 使得训练集合验证集中的数据标签的分布尽可能接近
                                                        random_state=42,
                                                        test_size=0.1, shuffle=True)
        return (Xtrain,Xtest,Ytrain,Ytest)

    # 加载模型
    def _load_model(self,path):
        print(path)
        if os.path.exists(path):
            return joblib.load(path)
        else:
            return False

    # 加载转换器
    def _load_transformer(self,path=None):

        if os.path.exists(path):
            joblib_tfv = joblib.load(path)
            return joblib_tfv
        else:
            return False

    # 加载XGBoost Encoding 模型参数
    def _load_w2v(self,path):
        xtrain_w2v = xvalid_w2v = None
        if os.path.exists(path):
            print("有诶")
            xtrain_w2v = np.load(path+"/xtrain_w2v.npy")
            xvalid_w2v = np.load(path+"/xvalid_w2v.npy")
        return (xtrain_w2v,xvalid_w2v)

    # 根据提供的转换器，处理数据集
    def _transform_training_data(self,joblib_tfv,Xtrain,Xtest):
        xtrain_tfv = joblib_tfv.transform(Xtrain)
        xvalid_tfv = joblib_tfv.transform(Xtest)
        return (xtrain_tfv,xvalid_tfv)

    # 创建Word2Vec编码后的numpy矩阵并保存(需要15分钟)
    def _build_training_w2v(self,Xtrain,Xtest):
        print("正在构建矩阵...")
        xtrain_w2v = [self._sent2vec(x) for x in tqdm(Xtrain)]
        xvalid_w2v = [self._sent2vec(x) for x in tqdm(Xtest)]
        np.save("./Data/Pretrained_Matrices/xtrain_w2v.npy",xtrain_w2v)
        np.save("./Data/Pretrained_Matrices/xvalid_w2v.npy",xvalid_w2v)
        return (xtrain_w2v,xvalid_w2v)

    # 准备LSTM数据，pad_sequence等等
    # 使用 keras tokenizer
    def _tokenize(self,max_len,Xtrain,Xtest):
        self.tokenizer = text.Tokenizer(num_words=None)

        self.tokenizer.fit_on_texts(list(Xtrain) + list(Xtest))
        xtrain_seq = self.tokenizer.texts_to_sequences(Xtrain)
        xvalid_seq = self.tokenizer.texts_to_sequences(Xtest)

        # 对文本序列进行zero填充
        xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
        xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

        word_index = self.tokenizer.word_index

        return word_index,xtrain_pad,xvalid_pad

    def _calculate_multiclass_logloss(self,actual, predicted, eps=1e-15):
        """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
        :param actual: 包含actual target classes的数组
        :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
        """
        # Convert 'actual' to a binary array if it's not already:
        if len(actual.shape) == 1:
            actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
            for i, val in enumerate(actual):
                actual2[i, val] = 1
            actual = actual2

        clip = np.clip(predicted, eps, 1 - eps)
        rows = actual.shape[0]
        vsota = np.sum(actual * np.log(clip))
        return -1.0 / rows * vsota

    def _calculate_recall(self,actual,predicted,eps=1e-15):
        if len(predicted.shape) != 1:
            clip = np.clip(predicted, eps, 1 - eps)
            predicted = np.argmax(clip,axis=1)

        rows = actual.shape[0]
        return np.sum(actual==predicted)/rows

##############################################################################
# 基础功能

    # 获取所有文章
    def extract_article(self,file=None):
        '''
        :param file:
        :return: 所有文章
        '''
        if not file:
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

    # def get_indexed_article(self,file):
    #     for i,article in enumerate(self.extract_article()):
    #


    # 获取所有文章标题
    def extract_title(self,file=None):
        '''

        :param file:
        :return: 所有标题
        '''
        if not file:
            file = self.speech
        lines = file.readlines()
        titles = set()
        for line in lines:
            if "'title'" in line:
                line = re.findall(r'\'title\':\s\'(.*)\'\,', line)
                if len(line) > 0:
                    if line[0] != '':
                        titles.add(line[0])

        return list(titles)

    # 找k个最相似的文章
    def find_k_most_similar_articles(self,id, k):
        '''
        用于语料库中文章相似度的对比
        :param id: 文章序号
        :param k: top k
        :return: None
        '''
        score_list = []
        for i in range(len(self.extracted_dictionary)):
            if i != id:
                score_list.append((i, self._document_distance(id, i)))

        score_list.sort(key=lambda x: x[1], reverse=True)

        print("This could take a while")

        print(f"The top {k} similar articles to {self.extracted_dictionary[id]['title']} are:\n")
        for i in range(k):
            print(self.extracted_dictionary[score_list[i][0]]["title"])

    # 计算文档距离
    def document_distance(self,id1, id2):
        '''
        返回两篇文章的相似度, 利用simHash算法
        :param id1: 第一篇文章的序号
        :param id2: 第二篇文章的序号
        :return: 文章的相似度
        '''
        info1 = self.extracted_dictionary[id1]
        info2 = self.extracted_dictionary[id2]

        article1 = info1["article"]
        article2 = info2["article"]

        article1 = self._jieba_clean_cut(article1)
        article2 = self._jieba_clean_cut(article2)

        hash1 = Simhash(article1)
        hash2 = Simhash(article2)

        return hash1.distance(hash2)

##############################################################################
# 项目驱动功能

    # 用word2vec给训练集编码, 用于需要w2v的模型训练使用
    def building_training_w2v(self):
        print("开始执行文本分词")
        X = self.training_data['文本分词']
        X = [i.split() for i in X]
        print(X)
        print("文本分词构建完毕")

        model = gensim.models.Word2Vec(X, min_count=5, window=8,
                                       vector_size=200)  # X是经分词后的文本构成的list，也就是tokens的列表的列表

        print("开始创建embeddings_index")
        embeddings_index = dict(zip(model.wv.index_to_key, model.wv.vectors))

        self.embedding = embeddings_index

        print('Found %s word vectors.' % len(embeddings_index))

        print("预处理训练数据")
        Xtrain, Xtest, Ytrain, Ytest = self._preprocess_training_data()

        print("开始创建pretrained_matrix")
        self._build_training_w2v(Xtrain, Xtest)

    # 建立word2vec的Embedding Matrix, 保存一个KeyedVectors对象
    def build_w2v_matrix(self):
        X = self.training_data['文本分词']
        X = [i.split() for i in X]

        model = gensim.models.Word2Vec(X, min_count=5, window=8,
                                       vector_size=200)  # X是经分词后的文本构成的list，也就是tokens的列表的列表
        embeddings_index = dict(zip(model.wv.index_to_key, model.wv.vectors))

        print('Found %s word vectors.' % len(model.wv))
        self.embedding = model.wv
        model.save("./Data/Pretrained_Matrices/embedding_dictionary.w2v")

    # 用于文章分类(已测试)
    def judge_topic_from_article(self,tf_idf_id, k):
        '''
        top_k 就是利用tf-idf求出的前k个与文章相关的行业相似度
        :param tf_idf_id: tf_idf中的文章序号
        :param k: top k
        :return: industry_score_mapping: 文章的行业相似度得分, top_k 前k个得分
        '''
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
                    if word in self.embedding:
                        similarity_list.append(self.embedding.similarity(word, industry))
                score = np.mean(similarity_list)
                industry_score_mapping[industry] = score

        top_k = dict(sorted(industry_score_mapping.items(), key=lambda x: x[1], reverse=True)[:k])

        return industry_score_mapping, top_k

    # 用于情感分析(已测试)
    def sentiment_from_summary(self,id, average=5):
        """
        :param id: 文章序号
        :param f: 文章路径
        :param average: 平均五个总结语句
        :return:
        """
        article = self.extracted_dictionary[id]["article"]
        title = self.extracted_dictionary[id]["title"]
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

    def get_tf_idf(self,id):
        article = self.extracted_dictionary[id]["article"]
        sa = SnowNLP(article)
        print(sa.tf)
        print(sa.idf)

    # LogisticRegression分类器(已测试)
    def predictByLogisticRegression(self,doc,use_pretrained=True):
        PRETRAINED_PATH = self.DEFAULT_PRETRAINED_MODEL_ROOT_PATH + "/LogisticRegression/LogisticRegression.pkl"
        PRETRAINED_TRANSFORMER = self.DEFAULT_PRETRAINED_MODEL_ROOT_PATH + "/LogisticRegression/Transformer.pkl"

        if not os.path.exists(self.DEFAULT_PROJECT_PATH+PRETRAINED_PATH) or not use_pretrained:
            print("正在重新训练")
            Xtrain, Xtest, Ytrain, Ytest = self._preprocess_training_data()
            stopWordList = [line.strip() for line in open(self.DEFAULT_STOP_WORDS_PATH, 'r', encoding='utf-8').readlines()]

            tfv = NLPNormalizeVectorizer(min_df=3,
                                         max_df=0.5,
                                         max_features=None,
                                         ngram_range=(1, 2),
                                         use_idf=True,
                                         smooth_idf=True,
                                         stop_words=stopWordList)

            # 使用TF-IDF来fit训练集和测试集, 这里使用将训练集和测试集一起进行特征TF-IDF提取
            tfv.fit(list(Xtrain) + list(Xtest))
            joblib_tfv = tfv

            xtrain_tfv,xvalid_tfv = self._transform_training_data(joblib_tfv,Xtrain,Xtest)

            # 利用提取的TFIDF特征来fit一个简单的Logistic Regression
            clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial',max_iter=5000)
            clf.fit(xtrain_tfv, Ytrain)

            joblib_model = clf
            joblib.dump(joblib_tfv,PRETRAINED_TRANSFORMER)
            joblib.dump(joblib_model,PRETRAINED_PATH)
        else:
            joblib_model = self._load_model(self.DEFAULT_PROJECT_PATH + PRETRAINED_PATH)
            joblib_tfv = self._load_transformer(self.DEFAULT_PROJECT_PATH + PRETRAINED_TRANSFORMER)

        print("使用预训练矩阵")
        doc = list(doc)
        print("正在转换")
        transformed = joblib_tfv.transform(doc)
        print("正在预测")
        Ypredict = joblib_model.predict(transformed)
        class_ = self.labelEncoder.inverse_transform(Ypredict)

        print(f"This article belongs to{class_}")

        return joblib_model.predict_proba(transformed)

    # SVM 分类器（已测试）
    def predictBySVM(self,doc,use_pretrained=True):
        PRETRAINED_PATH = self.DEFAULT_PRETRAINED_MODEL_ROOT_PATH + "/SVM/SVM.pkl"
        PRETRAINED_TRANSFORMER = self.DEFAULT_PRETRAINED_TRANSFORMER_ROOT_PATH + "/Transformer.pkl"
        PRETRAINED_TRANSFORMER_SVD = self.DEFAULT_PRETRAINED_MODEL_ROOT_PATH + "/SVM/SVD.pkl"
        PRETRAINED_TRANSFORMER_SCL = self.DEFAULT_PRETRAINED_MODEL_ROOT_PATH + "/SVM/SCL.pkl"

        Xtrain, Xtest, Ytrain, Ytest = self._preprocess_training_data()
        if not os.path.exists(PRETRAINED_PATH) or not use_pretrained:
            print("不使用预训练的模型")

            stopWordList = [line.strip() for line in open(self.DEFAULT_STOP_WORDS_PATH, 'r', encoding='utf-8').readlines()]

            tfv = NLPNormalizeVectorizer(min_df=3,
                                         max_df=0.5,
                                         max_features=None,
                                         ngram_range=(1, 2),
                                         use_idf=True,
                                         smooth_idf=True,
                                         stop_words=stopWordList)

            # 使用TF-IDF来fit训练集和测试集, 这里使用将训练集和测试集一起进行特征TF-IDF提取
            tfv.fit(list(Xtrain) + list(Xtest))
            joblib_tfv = tfv

            xtrain_tfv,xvalid_tfv = self._transform_training_data(joblib_tfv,Xtrain,Xtest)

            # SVM
            # 使用SVD进行降维，components设为120，对于SVM来说，SVD的components的合适调整区间一般为120~200
            svd = decomposition.TruncatedSVD(n_components=120)
            svd.fit(xtrain_tfv)
            xtrain_svd = svd.transform(xtrain_tfv)
            xvalid_svd = svd.transform(xvalid_tfv)

            # 对从SVD获得的数据进行缩放
            scl = preprocessing.StandardScaler()
            scl.fit(xtrain_svd)
            xtrain_svd_scl = scl.transform(xtrain_svd)
            xvalid_svd_scl = scl.transform(xvalid_svd)

            # 调用下SVM模型
            clf = SVC(C=1.0, probability=True)  # since we need probabilities
            clf.fit(xtrain_svd_scl, Ytrain)

            joblib_SVD = svd
            joblib_SCL = scl

            joblib_model = clf
            joblib.dump(joblib_tfv,PRETRAINED_TRANSFORMER)
            joblib.dump(joblib_model,PRETRAINED_PATH)
            joblib.dump(joblib_SVD,PRETRAINED_TRANSFORMER_SVD)
            joblib.dump(joblib_SCL,PRETRAINED_TRANSFORMER_SCL)
        else:
            joblib_tfv = self._load_transformer(PRETRAINED_TRANSFORMER)
            joblib_model = self._load_model(PRETRAINED_PATH)
            joblib_SVD = self._load_model(PRETRAINED_TRANSFORMER_SVD)
            joblib_SCL = self._load_model(PRETRAINED_TRANSFORMER_SCL)

        #正在使用预训练的模型
        print("正在使用预训练的模型")
        doc = list(doc)
        transformed = joblib_tfv.transform(doc)

        doc = joblib_SVD.transform(transformed)
        doc1 = joblib_SCL.transform(doc)

        class_ = joblib_model.predict(doc1)

        print(f"This article belongs to{self.labelEncoder.inverse_transform(class_)}")

        return joblib_model.predict_proba(doc1)

    # XGboost 分类器 （已测试）
    def predictByXGBoost(self,doc,encoding=False,use_pretrained=True):
        PRETRAINED_TF_IDF_PATH = self.DEFAULT_PRETRAINED_MODEL_ROOT_PATH + "/XGBoost/XGBoost.pkl"
        PRETRAINED_W2V_PATH = self.DEFAULT_PRETRAINED_MODEL_ROOT_PATH+"/XGBoost/XGBoost_w2v.pkl"
        PRETRAINED_TRANSFORMER = self.DEFAULT_PRETRAINED_TRANSFORMER_ROOT_PATH + "/Transformer.pkl"

        # 为了得到labelEncoder
        Xtrain, Xtest, Ytrain, Ytest = self._preprocess_training_data()


        # Not Recommended, too slow
        if self.training_embedding:
            if os.path.exists("./Data/Pretrained_Matrices/embedding_dictionary.w2v"):
                print("有预训练的embedding矩阵")
                self.embedding = gensim.models.word2vec.Word2Vec.load("./Data/Pretrained_Matrices/embedding_dictionary.w2v").wv
                # print(self.embedding)
            else:
                X = self.training_data['文本分词']
                X = [i.split() for i in X]

                model = gensim.models.Word2Vec(X, min_count=5, window=8,
                                               vector_size=200)  # X是经分词后的文本构成的list，也就是tokens的列表的列表
                # embeddings_index = dict(zip(model.wv.index_to_key, model.wv.vectors))

                print('Found %s word vectors.' % len(model.wv))
                self.embedding = model.wv

        # 基于TF-IDF (已测试)
        if not encoding:

            print("基于TF-IDF的XGBoost")
            # xgboost 时间有点久
            if not os.path.exists(PRETRAINED_TF_IDF_PATH) or not use_pretrained:
                print("正在重新训练")
                stopWordList = [line.strip() for line in
                                open(self.DEFAULT_STOP_WORDS_PATH, 'r', encoding='utf-8').readlines()]

                tfv = NLPNormalizeVectorizer(min_df=3,
                                             max_df=0.5,
                                             max_features=None,
                                             ngram_range=(1, 2),
                                             use_idf=True,
                                             smooth_idf=True,
                                             stop_words=stopWordList)

                # 使用TF-IDF来fit训练集和测试集, 这里使用将训练集和测试集一起进行特征TF-IDF提取
                tfv.fit(list(Xtrain) + list(Xtest))
                joblib_tfv = tfv

                xtrain_tfv, xvalid_tfv = self._transform_training_data(joblib_tfv, Xtrain, Xtest)

                clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                                        subsample=0.8, nthread=10, learning_rate=0.1)

                print("开始训练")
                clf.fit(xtrain_tfv.tocsc(), Ytrain)

                joblib_model = clf

                joblib.dump(joblib_tfv, PRETRAINED_TRANSFORMER)

                joblib.dump(joblib_model, PRETRAINED_TF_IDF_PATH)
            else:
                joblib_model = self._load_model(PRETRAINED_TF_IDF_PATH)
                joblib_tfv = self._load_transformer(PRETRAINED_TRANSFORMER)

            doc = list(doc)
            # print("doc")
            # print(doc)
            transformed = joblib_tfv.transform(doc)
            # print("transformed")
            # print(transformed)
            Ypredict = joblib_model.predict(transformed.tocsc())
            print(f"TF-IDF-XGBoost模型预测的结果是{self.labelEncoder.inverse_transform(Ypredict)}")

            return joblib_model.predict_proba(transformed.tocsc())
        # 基于WordVec
        else:
            print("基于word2vec的XGBoost")
            # 对训练集和验证集使用上述函数，进行文本向量化处理，速度非常慢, 推荐预训练
            xtrain_w2v, xvalid_w2v = self._load_w2v(self.DEFAULT_PRETRAINED_MATRICES_ROOT_PATH)
            joblib_model = self._load_model(PRETRAINED_W2V_PATH)
             # 未完待续
            if not os.path.exists(PRETRAINED_W2V_PATH) or not use_pretrained:
                print("不使用预训练XGBoost模型")
                joblib_model = xgb.XGBClassifier(nthread=2,silent=False)
                joblib_model.fit(xtrain_w2v,Ytrain)
                joblib.dump(joblib_model,PRETRAINED_W2V_PATH)
            else:
                print("有XGBoost模型")
                joblib_model = self._load_model(PRETRAINED_W2V_PATH)
                # if not type(xtrain_w2v)==np.ndarray or not type(xvalid_w2v)==np.ndarray:
                #     print("没有训练完的xtrain_w2v")
                #     xtrain_w2v = [self._sent2vec(x) for x in Xtrain]
                #     xvalid_w2v = [self._sent2vec(x) for x in Xtest]

            doc_w2v = np.array([self._sent2vec(x) for x in doc])
            Ypredict = joblib_model.predict(doc_w2v)
            class_ = self.labelEncoder.inverse_transform(Ypredict)

            print(f"基于Word2Vec的模型预测结果是{class_}")
            return joblib_model.predict_proba(doc_w2v)

    # 常规神经网络（已测试）
    def predictByDense(self,doc=None,use_pretrained=True):
        # Not Recommended, too slow (About 40 min)

        if self.training_embedding:
            if os.path.exists("./Data/Pretrained_Matrices/embedding_dictionary.w2v"):
                print("有预训练的embedding矩阵")
                self.embedding = gensim.models.word2vec.Word2Vec.load("./Data/Pretrained_Matrices/embedding_dictionary.w2v").wv
                # print(self.embedding)
            else:
                X = self.training_data['文本分词']
                X = [i.split() for i in X]

                model = gensim.models.Word2Vec(X, min_count=5, window=8,
                                               vector_size=200)  # X是经分词后的文本构成的list，也就是tokens的列表的列表
                # embeddings_index = dict(zip(model.wv.index_to_key, model.wv.vectors))

                print('Found %s word vectors.' % len(model.wv))
                self.embedding = model.wv

        Xtrain, Xtest, Ytrain, Ytest = self._preprocess_training_data()

        xtrain_w2v,xvalid_w2v = self._load_w2v(self.DEFAULT_PRETRAINED_MATRICES_ROOT_PATH)

        if type(xtrain_w2v)!=np.ndarray or type(xvalid_w2v)!=np.ndarray:
            # Takes about 15 minutes
            xtrain_w2v, xvalid_w2v = self._build_training_w2v(Xtrain, Xtest)

        if not os.path.exists("./Data/Pretrained_Model/DenseNeural/Dense300.h5") or not use_pretrained:
            print("不使用Pretrained")
            # Deep Learning
            # 对数据进行标准化缩放用于传入神经网络
            scl = preprocessing.StandardScaler()
            xtrain_w2v_scl = scl.fit_transform(xtrain_w2v)
            xvalid_w2v_scl = scl.transform(xvalid_w2v)

            # joblib.dump("./Data/Pretrained_Model/DenseNeural/DenseSCL.pkl",scl)

            # 对标签进行binarize处理
            #ytrain_enc是一个(batch_size,19)的结果矩阵
            ytrain_enc = np_utils.to_categorical(Ytrain)
            yvalid_enc = np_utils.to_categorical(Ytest)

            doc_scl = xvalid_w2v_scl

            # 创建1个3层的序列神经网络（Sequential Neural Net）
            # print(xtrain_w2v_scl.shape)
            # print(ytrain_enc.shape)
            model = Sequential()

            model.add(Dense(300, input_dim=200, activation='relu'))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())

            model.add(Dense(300, activation='relu'))
            model.add(Dropout(0.3))
            model.add(BatchNormalization())

            model.add(Dense(19)) #分成19类
            model.add(Activation('softmax'))

            # 模型编译
            model.compile(loss='categorical_crossentropy', optimizer='adam')
            print(model.summary())

            model.fit(xtrain_w2v_scl, y=ytrain_enc, batch_size=64,
                      epochs=15, verbose=1,
                      validation_data=(xvalid_w2v_scl, yvalid_enc))
            model.save("./Data/Pretrained_Model/DenseNeural/Dense300.h5")
        else:
            print("使用预训练的Dense模型")
            model = keras.models.load_model("./Data/Pretrained_Model/DenseNeural/Dense300.h5")
            # scl = joblib.load("./Data/Pretrained_Model/DenseNeural/DenseSCL.pkl")
            scl = preprocessing.StandardScaler()

            # 先将输入的文本转化成向量表示形式
            doc_w2v = [self._sent2vec(x) for x in doc]
            # print(doc_w2v)
            doc_scl = scl.fit_transform(doc_w2v)


        Ypredict = model.predict(doc_scl)
        Ypredict_dummy = np.argmax(Ypredict,axis=1)
        Ypredict_label = self.labelEncoder.inverse_transform(Ypredict_dummy)
        print("批量模型预测结果")
        print(Ypredict_label)
        return Ypredict

    # LSTM神经网络 (已测试)
    def predictByLSTM(self,doc=None,padding_max_len=50,layer_num=2,bidirection=False,encoding=False,use_pretrained=True):
        if self.training_embedding:
            if os.path.exists("./Data/Pretrained_Matrices/embedding_dictionary.w2v"):
                print("有预训练的embedding矩阵")
                self.embedding = gensim.models.word2vec.Word2Vec.load(
                    "./Data/Pretrained_Matrices/embedding_dictionary.w2v").wv
            else:
                X = self.training_data['文本分词']
                X = [i.split() for i in X]

                model = gensim.models.Word2Vec(X, min_count=5, window=8,
                                               vector_size=200)  # X是经分词后的文本构成的list，也就是tokens的列表的列表
                # embeddings_index = dict(zip(model.wv.index_to_key, model.wv.vectors))
                #
                # print('Found %s word vectors.' % len(embeddings_index))
                self.embedding = model.wv

        Xtrain,Xtest,Ytrain,Ytest = self._preprocess_training_data()

        # 对标签进行binarize处理, 处理成一个第二维度为19的embedding matrix, (sample_number,class_nums) with 1 representing the class index
        ytrain_enc = np_utils.to_categorical(Ytrain)
        yvalid_enc = np_utils.to_categorical(Ytest)



        word_index,xtrain_pad,xtest_pad = self._tokenize(padding_max_len,Xtrain,Xtest)

        # 基于已有的数据集中的词汇创建一个词嵌入矩阵（Embedding Matrix）
        print("正在创建Embedding Matrix")
        embedding_matrix = np.zeros((len(word_index) + 1, 200))
        embedding_dict = dict(zip(self.embedding.index_to_key,self.embedding.vectors))
        for word, i in tqdm(word_index.items()):
            embedding_vector = embedding_dict.get(word,None)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        print("Embedding Matrix 创建完毕")


        # 基于前面训练的Word2vec词向量，使用1个两层的LSTM模型
        if not bidirection:
            if not os.path.exists("./Data/Pretrained_Model/LSTM/LSTM_simple.h5") or not use_pretrained:
                model = Sequential()
                model.add(Embedding(len(word_index) + 1,
                                    200,
                                    weights=[embedding_matrix],
                                    input_length=padding_max_len,
                                    trainable=False))
                model.add(SpatialDropout1D(0.3))
                model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

                model.add(Dense(1024, activation='relu'))
                model.add(Dropout(0.8))

                model.add(Dense(1024, activation='relu'))
                model.add(Dropout(0.8))

                model.add(Dense(19))
                model.add(Activation('softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam')

                model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, verbose=1,
                          validation_data=(xtest_pad, yvalid_enc))
                model.save("./Data/Pretrained_Model/LSTM/LSTM_simple.h5")
            else:
                print("使用预训练的LSTM-simple模型")
                model = keras.models.load_model("./Data/Pretrained_Model/LSTM/LSTM_simple.h5")
        else:
            # 基于前面训练的Word2vec词向量，构建1个2层的Bidirectional LSTM
            if not os.path.exists("./Data/Pretrained_Model/LSTM/LSTM_Bi.h5") or not use_pretrained:
                model = Sequential()
                model.add(Embedding(len(word_index) + 1,
                                    200,
                                    weights=[embedding_matrix],
                                    input_length=padding_max_len,
                                    trainable=False))
                model.add(SpatialDropout1D(0.3))
                model.add(Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3)))

                model.add(Dense(1024, activation='relu'))
                model.add(Dropout(0.8))

                model.add(Dense(1024, activation='relu'))
                model.add(Dropout(0.8))

                model.add(Dense(19))
                model.add(Activation('softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam')

                # 在模型拟合时，使用early stopping这个回调函数防止过拟合
                # monitor就是监视指标，min_delta最小提升度，就是一个batch下来val_loss至少需要下降min_delta这么多才能算是有效gradient_descent
                # patience就是当无效迭代到达patience时停止训练, mode='auto' 自动判断优化方向
                earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
                model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100,
                          verbose=1, validation_data=(xtest_pad, yvalid_enc), callbacks=[earlystop])
                model.save("./Data/Pretrained_Model/LSTM/LSTM_Bi.h5")

            else:
                print("使用预训练的Bi-LSTM模型")
                model = keras.models.load_model("./Data/Pretrained_Model/LSTM/LSTM_Bi.h5")

        #创造预测接口

        doc_seq = self.tokenizer.texts_to_sequences(doc)

        # 对文本序列进行zero填充
        doc_pad = sequence.pad_sequences(doc_seq, maxlen=padding_max_len)

        Ypredict = model.predict(doc_pad)
        Ypredict_dummy = np.argmax(Ypredict, axis=1)
        Ypredict_label = self.labelEncoder.inverse_transform(Ypredict_dummy)
        print("批量模型预测结果")
        print(Ypredict_label)
        return Ypredict

    # 以上所有模型的糅合 (还未测试)
    def predictByModelEnsembling(self):
        pass

#####################################################################################
# 评估方法

    def generate_random(self,seed=42,sample_number=2000):
        self.random_index = np.random.choice(self.training_data.shape[0],sample_number)
        self.testSample = list(self.training_data.iloc[self.random_index, :]["文本分词"])
        self.actualLabel = self.labels[self.random_index]

    def eva_logit(self):
        Ypredict = self.predictByLogisticRegression(self.testSample,use_pretrained=True)
        print(f"逻辑回归的cross entropy loss是:{self._calculate_multiclass_logloss(self.actualLabel,Ypredict)}")
        print(f"逻辑回归的recall是:{self._calculate_recall(self.actualLabel,Ypredict)}")

    def eva_SVM(self):
        Ypredict = self.predictBySVM(self.testSample,use_pretrained=True)
        print(f"SVM的cross entropy loss是:{self._calculate_multiclass_logloss(self.actualLabel,Ypredict)}")
        print(f"SVM的recall是:{self._calculate_recall(self.actualLabel,Ypredict)}")

    def eva_XGBoost(self,w2v=False):
        if w2v:
            Ypredict = self.predictByXGBoost(self.testSample,encoding=True,use_pretrained=True)
            print(f"XGB_W2V的cross entropy loss是:{self._calculate_multiclass_logloss(self.actualLabel,Ypredict)}")
            print(f"XGB_W2V的recall是:{self._calculate_recall(self.actualLabel,Ypredict)}")
        else:
            # 有一些问题
            Ypredict = self.predictByXGBoost(self.testSample,encoding=False,use_pretrained=True)
            print(f"XGB_TF_IDF的cross entropy loss是:{self._calculate_multiclass_logloss(self.actualLabel,Ypredict)}")
            print(f"XGB_TF_IDF的recall是:{self._calculate_recall(self.actualLabel,Ypredict)}")

    def eva_Dense(self):
        Ypredict = self.predictByDense(self.testSample,use_pretrained=True)
        print(f"Dense的cross entropy loss是:{self._calculate_multiclass_logloss(self.actualLabel,Ypredict)}")
        print(f"Dense的recall是:{self._calculate_recall(self.actualLabel,Ypredict)}")

    def eva_LSTM(self,bi=False):
        if bi:
            Ypredict = self.predictByLSTM(self.testSample, bidirection=True, use_pretrained=True)
            print(f"LSTM的cross entropy loss是:{self._calculate_multiclass_logloss(self.actualLabel,Ypredict)}")
            print(f"LSTM的recall是:{self._calculate_recall(self.actualLabel,Ypredict)}")
        else:
            Ypredict = self.predictByLSTM(self.testSample, bidirection=False, use_pretrained=True)
            print(f"LSTM的cross entropy loss是:{self._calculate_multiclass_logloss(self.actualLabel,Ypredict)}")
            print(f"LSTM的recall是:{self._calculate_recall(self.actualLabel,Ypredict)}")

    def eva_Ensemble(self):
        pass

#####################################################################################
# 测试方法(已测试)

    # 逻辑回归测试
    def logisticTest(self):
        testSample = list(self.training_data["文本分词"])[0:1]
        predicted = self.predictByLogisticRegression(testSample,use_pretrained=False)
        print(f"类别是{self.labelEncoder.inverse_transform(predicted)}")


    def SVMTest(self):
        testSample = list(self.training_data["文本分词"])[0:2]
        predicted = self.predictBySVM(testSample,True)
        print(f"类别是{self.labelEncoder.inverse_transform(predicted)}")


    def testUtils(self):
        # xtrain_w2v, xvalid_w2v = self._load_w2v(self.DEFAULT_PRETRAINED_MATRICES_ROOT_PATH)
        # print(type(xtrain_w2v)==np.ndarray)
        # print(xvalid_w2v.sum())
        # print(xtrain_w2v.sum())
        word2vec = gensim.models.word2vec.Word2Vec.load("./Data/Pretrained_Matrices/embedding_dictionary.w2v").wv
        # print(embedding_dictionary)
        print(type(word2vec))
        print(word2vec.index_to_key)
        print(len(word2vec.vectors[1]))


    def testXGBoost(self,w2v = True,pretrained=True):
        testSample = list(self.training_data["文本分词"])[0:1]
        if w2v:
            self.predictByXGBoost(testSample,encoding=True,use_pretrained=pretrained)
        else:
            self.predictByXGBoost(testSample,encoding=False,use_pretrained=pretrained)


    def testDense(self,use_pretrained):
        testSample = list(self.training_data["文本分词"])[5:10]
        # testSample = list(map(lambda lissy:lissy.split(' '),testSample))
        # print(testSample)
        print(self.predictByDense(testSample,use_pretrained))


    def testLSTM(self):
        testSample = list(self.training_data["文本分词"])[0:1]
        print(self.predictByLSTM(testSample,bidirection=True))


if __name__ == "__main__":
    nlp = NLP_Text_Analysis(training_embedding=False,load_embedding=False,training_model=False)
    nlp.get_tf_idf(5)
