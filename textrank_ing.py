#-*- coding:utf-8 -*-

from konlpy.tag import Kkma
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from datetime import datetime


class SentenceTokenizer(object):
    def __init__(self):
        self.twitter = Twitter()
        self.stopwords = []


    def text2sentences(self, file_name):
        ktlk_csv = pd.read_csv(file_name , encoding="utf-8-sig")
        df = pd.DataFrame(ktlk_csv)

        nouns = []
        for sentence in df["Message"]:
            nouns.append(" ".join([noun for noun in self.twitter.nouns(sentence)
                                    if noun not in self.stopwords and len(noun) > 1]))

        nouns = [item for item in nouns
                if item != '']
        
        return nouns

class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence
    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}


class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
        if link_sum != 0:
            A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
            B = (1-d) * np.ones((matrix_size, 1))
            ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}


class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()

        self.sentences = self.sent_tokenize.text2sentences(text)
        
#        self.nouns = self.sent_tokenize.get_nouns(self.sentences) # nouns 를 받고.
                    
        self.graph_matrix = GraphMatrix() # 그래프 만들기. 
        self.sent_graph = self.graph_matrix.build_sent_graph(self.sentences) # 문장그래프 
#        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns) # 단어그래프. 
        
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
        
        self.word_rank_idx =  self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
        
        
    def summarize(self, sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        
        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])
        
        return summary
        
    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)
        
        keywords = []
        index=[]
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)
            
        #index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])
        
        return keywords

textrank=TextRank("beautig.csv")
for row in textrank.summarize(3):
    print(row)
    print()
    print('keywords :',textrank.keywords())
