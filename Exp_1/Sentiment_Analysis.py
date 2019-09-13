# file read
import os
import nltk
import math


# 利用贝叶斯公式计算属于两种类别的可能性，选取可能性大的那个作为结果
def bayes(input_txt):
    possibility_pos = 0
    possibility_neg = 0
    for word in input_txt:
        if word in pos_word_dic.keys() and word in neg_word_dic.keys():
            # 因为两个label的训练数据数量相同，因此不需要乘label的概率。使用log加快计算并防止下溢
            possibility_pos += math.log(pos_freq_dic[word])
            possibility_neg += math.log(neg_freq_dic[word])
            # print('poss_pos: ', possibility_pos)
            # print('poss_neg: ', possibility_neg)
        else:
            continue
    if possibility_pos > possibility_neg:
        return 1
    else:
        return 0


# nltk.download('all')
path = r'D:\社交舆情实验\Experiment_for_Social_Network-master\Experiment_for_Social_Network-master\Experiment_1\1_Sentiment_Analysis\aclImdb_v1\aclImdb\train'
# path = r'D:\社交舆情实验\Exp_1\toy_train'
pos_path = path + '/pos/'
neg_path = path + '/neg/'

test_path = r'D:\社交舆情实验\Experiment_for_Social_Network-master\Experiment_for_Social_Network-master\Experiment_1\1_Sentiment_Analysis\aclImdb_v1\aclImdb\test'
# test_path = r'D:\社交舆情实验\Exp_1\toy_test'
test_pos_path = test_path + '/pos/'
test_neg_path = test_path + '/neg/'

print('reading texts......')
# 读取文件
pos_files = [pos_path + x for x in filter(lambda x: x.endswith('.txt'), os.listdir(pos_path))]
neg_files = [neg_path + x for x in filter(lambda x: x.endswith('.txt'), os.listdir(neg_path))]
test_pos_files = [test_pos_path + x for x in filter(lambda x: x.endswith('.txt'), os.listdir(test_pos_path))]
test_neg_files = [test_neg_path + x for x in filter(lambda x: x.endswith('.txt'), os.listdir(test_neg_path))]

pos_list = [open(x, 'r', encoding='utf-8').read().lower() for x in pos_files]
neg_list = [open(x, 'r', encoding='utf-8').read().lower() for x in neg_files]
test_pos_list = [open(x, 'r', encoding='utf-8').read().lower() for x in test_pos_files]
test_neg_list = [open(x, 'r', encoding='utf-8').read().lower() for x in test_neg_files]

data_list = pos_list + neg_list
labels_list = [1] * len(pos_list) + [0] * len(neg_list)
test_data_list = test_pos_list + test_neg_list
test_labels_list = [1] * len(test_pos_list) + [0] * len(test_neg_list)

# 对文本进行分词
tokenized_pos_list = [nltk.word_tokenize(x) for x in pos_list]
tokenized_neg_list = [nltk.word_tokenize(x) for x in neg_list]

tokenized_test_list = [nltk.word_tokenize(x) for x in test_data_list]
# 建立情感词词典
f = open('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r', encoding='utf-8')
txt = f.read().split('\n')
emotion_word = []
# print(len(txt))
for line in txt:
    # print(line)
    word = line.split('\t')[0]
    emotion = line.split('\t')[1]
    label = line.split('\t')[2]
    if label == '1':
        emotion_word.append(word)
emotion_word = list(set(emotion_word))
# print('emotional word number: ', len(emotion_word))
# print('emotion_word: ', emotion_word)
clean_pos_list = []
clean_neg_list = []
# 去除非情感词
for line in tokenized_pos_list:
    new_line = []
    for word in line:
        if word in emotion_word:
            new_line.append(word)
    clean_pos_list.append(new_line)

for line in tokenized_neg_list:
    new_line = []
    for word in line:
        if word in emotion_word:
            new_line.append(word)
    clean_neg_list.append(new_line)

# 统计pos/neg总词数和各个词的词频
count_pos_word = 0
count_neg_word = 0
pos_word_dic = {}
neg_word_dic = {}

for line in clean_pos_list:
    for word in line:
        if word not in pos_word_dic.keys():
            pos_word_dic[word] = 1
        else:
            pos_word_dic[word] += 1
for key in pos_word_dic.keys():
    count_pos_word += pos_word_dic[key]

for line in clean_neg_list:
    for word in line:
        if word not in neg_word_dic.keys():
            neg_word_dic[word] = 1
        else:
            neg_word_dic[word] += 1
for key in neg_word_dic.keys():
    count_neg_word += neg_word_dic[key]

# print(count_neg_word)
# print(count_pos_word)

# 计算在不同的类别中每个词的可能性
pos_freq_dic = {}
neg_freq_dic = {}
for key in pos_word_dic.keys():
    pos_freq_dic[key] = pos_word_dic[key] / count_pos_word
for key in neg_word_dic.keys():
    neg_freq_dic[key] = neg_word_dic[key] / count_neg_word

# 用贝叶斯计算可能性
correct = 0
for i in range(len(test_data_list)):
    if bayes(tokenized_test_list[i]) == test_labels_list[i]:
        correct += 1
print('total: ', len(test_data_list), ', correct: ', correct, ', accuracy: ', correct/len(test_data_list))





















