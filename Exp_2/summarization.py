from load_data import load_stories
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
from nltk.corpus import stopwords
import numpy as np
import math
# load data
directory = r'D:\社交舆情实验\Experiment_for_Social_Network-master\Experiment_for_Social_Network-master\Experiment_1\2_automatic_summarization\data\cnn_stories_tokenized\\'
stories = load_stories(directory, 5)
print('Loaded Stories %d' % len(stories))
ref = []
for index, story in enumerate(stories):
    ref.append(story['highlights'])
    txt = story['story']
    sents = txt.split('\n')
    for i in range(len(sents)):
        if '' in sents:
            sents.remove('')
        else:
            break
    for i in range(len(sents)):
        trans = str.maketrans({key: None for key in string.punctuation})
        sents[i] = sents[i].translate(trans)
    tokenized_sents = [nltk.word_tokenize(x) for x in sents]
    words = stopwords.words('english')
    clean_tokenized_sents = [[w for w in line if w not in words] for line in tokenized_sents]
    # 统计DF
    txt_word = []
    for line in clean_tokenized_sents:
        for word in line:
            if word not in txt_word:
                txt_word.append(word)
    doc_freq = {}
    word_index = {}
    for ind, word in enumerate(txt_word):
        doc_freq[word] = 0
        word_index[word] = ind
    for word in txt_word:
        for line in clean_tokenized_sents:
            if word in line:
                doc_freq[word] += 1
    word_number = len(txt_word)
    word_idf = {}
    for word in txt_word:
        word_idf[word] = 1 / math.log(doc_freq[word] + 1)
    vec_words_dic = {}
    for word in txt_word:
        vec_word = np.zeros(word_number)
        vec_word[word_index[word]] = word_idf[word]
        vec_words_dic[word] = vec_word
    
    vec_sents_dic = {}
    for i in range(len(clean_tokenized_sents)):
        vec_sents_dic[i] = np.zeros(word_number)
        for word in clean_tokenized_sents[i]:
            vec_sents_dic[i] += vec_words_dic[word]
            
    similarity_matrix = np.zeros((len(clean_tokenized_sents), len(clean_tokenized_sents)))
    for i in range(len(clean_tokenized_sents)):
        for j in range(len(clean_tokenized_sents)):
            similarity_matrix[i][j] = cosine_similarity(vec_sents_dic[i].reshape(1,-1), vec_sents_dic[j].reshape(1,-1))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    