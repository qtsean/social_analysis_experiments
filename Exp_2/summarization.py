from load_data import load_stories
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
from nltk.corpus import stopwords
import numpy as np
import math
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
# load data
data_number = 200
directory = r'D:\社交舆情实验\Experiment_for_Social_Network-master\Experiment_for_Social_Network-master\Experiment_1\2_automatic_summarization\data\cnn_stories_tokenized\\'
stories = load_stories(directory, data_number)
print('Loaded Stories %d' % len(stories))
ref = []
ans = []
bleu_score = 0
rouge_1 = 0
rouge_2 = 0
rouge_l = 0
rouge = RougeCalculator(stopwords=True, lang="en")
bleu = BLEUCalculator()
empty = 0
count_max_0 = 0
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
    if len(clean_tokenized_sents) == 0:
        empty += 1
        ans.append('')
        continue
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
    
    # R = MR
    M = similarity_matrix.copy()
    for i in range(len(clean_tokenized_sents)):
        M[i][i] = 0
        
    for j in range(len(clean_tokenized_sents)):
        Sum = 0
        for i in range(len(clean_tokenized_sents)):
            Sum += M[i][j]
        if Sum == 0:
            for i in range(len(clean_tokenized_sents)):
                M[i][j] = 1 / len(clean_tokenized_sents)
        else:
            for i in range(len(clean_tokenized_sents)):
                M[i][j] /= Sum
    d = 0.95
#    print(M)
    M = (1 - d) * np.ones_like(M) + d * M
#    print(M)
    R = np.ones(len(clean_tokenized_sents)) / len(clean_tokenized_sents)
    R.reshape((-1, 1))
    # print(R.shape)
    for i in range(100):
        # print(i)
        currentR = R
        R = np.dot(M, R)
        if abs((currentR - R).sum()) < 0.00001:
            break
    R[0] *= 3
    R[-1] *= 3
    max_index = np.argmax(R)
    if max_index == 0:
        count_max_0 += 1
    ans.append(sents[max_index])
    
    print(index)
#    print(ref[index])
#    print(ans[index])
    rouge_1 += rouge.rouge_n(summary=ans[index],
                             references=ref[index],
                             n=1)
    rouge_2 += rouge.rouge_n(summary=ans[index],
                             references=ref[index],
                             n=2)
    rouge_l += rouge.rouge_l(
            summary=ans[index],
            references=ref[index])
    bleu_score += bleu.bleu(ans[index], ref[index][0])
   
data_number -= empty
print('average rouge_1 = ', rouge_1/data_number)
print('average rouge_2 = ', rouge_2/data_number)
print('average rouge_L = ', rouge_l/data_number) 
print('average bleu = ', bleu_score/data_number)   
print('选择第一句的概率: ', count_max_0/data_number)

rouge_1 = 0
rouge_2 = 0
rouge_l = 0
bleu_score = 0
from gensim.summarization.summarizer import summarize
count = 0
for story in stories:
    txt = story['story']
    sents = nltk.sent_tokenize(txt)
    if len(sents) < 12:
        continue
    summary = summarize(txt, ratio = 0.1, split = True)
    ref = story['highlights']
#    print('summary: ', summary[0])
#    print('ref: ', ref[0])
    rouge_1 += rouge.rouge_n(summary=summary[0],
                             references=ref[0],
                             n=1)
    rouge_2 += rouge.rouge_n(summary=summary[0],
                             references=ref[0],
                             n=2)
    rouge_l += rouge.rouge_l(
                            summary=summary[0],
                             references=ref[0])
    bleu_score += bleu.bleu(summary[0], ref[0])
    
    
    
    count += 1
#    print(count)
    
print('\nbaseline_average rouge_1 = ', rouge_1/count)
print('baseline_average rouge_2 = ', rouge_2/count)
print('baseline_average rouge_L = ', rouge_l/count) 
print('baseline_average bleu = ', bleu_score/count)   
    
    
    
    
    
    
    