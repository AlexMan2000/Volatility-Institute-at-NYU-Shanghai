import pandas as pd
import gensim
from gensim.parsing.preprocessing import preprocess_documents
from gensim.models.doc2vec import Doc2Vec,TaggedDocument


class Document2Vector:

    DEFAULT_MODEL_SAVE_PATH = "E:\VINS Project\Data\Pretrained_Model\doc2vec\doc2vecModel.d2v"

    def __init__(self,vector_size=200,min_count=1,epochs=50):
        self.model = Doc2Vec(vector_size=vector_size,min_count=min_count,epochs=epochs)


    def read_corpus(self,doc_list):
        for sub_dictionary in doc_list:
            yield TaggedDocument(gensim.utils.to_unicode(sub_dictionary["article"]).split(" "),sub_dictionary["title"])


    def build_vocab(self,doc_list):
        self.train_corpus = list(self.read_corpus(doc_list))

        print("建立模型中")
        self.model.build_vocab(self.train_corpus)

    def train_model(self):
        for epoch in range(1):
            self.model.train(self.train_corpus,total_examples=self.model.corpus_count,epochs=self.model.iter)

    def save_model(self):
        self.model.save(self.DEFAULT_MODEL_SAVE_PATH)