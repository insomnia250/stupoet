#coding=utf-8
from functools import reduce
import re
import numpy as np
import codecs
import os
# from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent,Dropout,Dense,RepeatVector
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import cPickle

def get_pairs_from_poems(file):
	pairs = []
	charNum = 0
	for i,line in enumerate(file):
		# if i==40:break
		content = line.strip('\n').split('|')[-1]
		poem = list(content.decode('utf8'))  #按utf8分词
		if (len(poem)>=12): 
			if (poem[5] in [u'，',u'。',u'！',u'？',u'；']) & (poem[11] in [u'，',u'。',u'！',u'？',u'；']):
				delta = 6
				pad = 1
		if (len(poem)>=16):
			if (poem[7] in [u'，',u'。',u'！',u'？',u'；']) & (poem[15] in [u'，',u'。',u'！',u'？',u'；']):
				delta = 8
				pad = 0
		else:
			continue
		pairCnt = int(len(poem)/(2*delta))
		for pairIdx in range(pairCnt):
			pair1 =  poem[pairIdx*2*delta : (pairIdx*2+1)*delta-1]
			pair2 = poem[(pairIdx*2+1)*delta : (pairIdx*2+2)*delta-1]
			if (poem[(pairIdx*2+1)*delta-1] in [u'，',u'。',u'！',u'？',u'；']) & \
				(poem[(pairIdx*2+2)*delta-1] in [u'，',u'。',u'！',u'？',u'；']):
				charNum += len(pair1 + pair2)
				pairs.append(pair1 + pair2)
	return pairs,charNum

def vectorize_stories(parsed_poem, word_to_index, poem_maxlen):
    xs = []
    for poem in parsed_poem:
        x = [word_to_index[w] for w in poem]
        xs.append(x)
    return xs
    # return pad_sequences(xs, maxlen=poem_maxlen ,padding='post',truncating='post')

def characterize(vec_poems, index_to_word):
    xs = []
    for poem in vec_poems:
        x = [index_to_word[idx] for idx in poem]
        xs.append(x)
    return xs



unknown_token = u'unknown'
NA_token = u'NA'
start_token = u'start'
end_token = u'end'

EMBEDDING_units = 50
LSTM1_cells = 50
LSTM2_cells = 50

file1 = open('../data/QuanTangShi/QuanTangShi.txt','rU')
file2 = open('../data/QuanSongShi/QuanSongShi.txt','rU')


#切分成单字，存在parsed_poem中[ ['君','不','见','，',...] , ['山','不','在',...] ]
pairs1,charNum1 = get_pairs_from_poems(file1)
pairs2,charNum2 = get_pairs_from_poems(file2)
pairs = pairs1+pairs2
charNum = charNum1+charNum2
del pairs1,pairs2,charNum1,charNum2
print 'total chars number:' , charNum
print 'total pairs number:' , len(pairs)

# 读词频文件：
vocab_freq = []
file = open('freq.txt','rU')
for i,line in enumerate(file):
	if i>=5500:break
	char = line.split(':')[0].decode('utf8')
	cnt = line.strip().split(':')[1]
	vocab_freq.append( (char,cnt) )

# word 和 int 映射
index_to_word = [x[0] for x in vocab_freq]
if NA_token in index_to_word:
	index_to_word.remove(NA_token)
index_to_word = [NA_token] + index_to_word
index_to_word.append(unknown_token)

word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
vocab_size = len(word_to_index)
print 'vocab size(with {0}(0), {1}({2}):{3}'\
		.format(NA_token, unknown_token,vocab_size-1, vocab_size)

#用u'unknown'代替低频词
for i, pair in enumerate(pairs):
    pairs[i] = [w if w in word_to_index else unknown_token for w in pair]


# bigram12 = {}
# bigram1 = {}
# for i,pair in enumerate(pairs):
# 	print i
# 	senten1 = [start_token] + pair[0:len(pair)/2] + [end_token]
# 	senten2 = [start_token] + pair[len(pair)/2:] + [end_token]

# 	for idx in range(len(senten1)-1):
# 		key = u'{0},{1}'.format(senten1[idx],senten1[idx+1])
# 		bigram12.setdefault(key, 0)
# 		bigram12[key]+=1
# 		bigram1.setdefault(senten1[idx], 0)
# 		bigram1[senten1[idx]]+=1
# 	bigram1.setdefault(senten1[idx+1], 0)
# 	bigram1[senten1[idx+1]]+=1

# 	for idx in range(len(senten2)-1):
# 		key = u'{0},{1}'.format(senten2[idx],senten2[idx+1])
# 		bigram12.setdefault(key, 0)
# 		bigram12[key]+=1
# 		bigram1.setdefault(senten2[idx], 0)
# 		bigram1[senten2[idx]]+=1
# 	bigram1.setdefault(senten2[idx+1], 0)
# 	bigram1[senten2[idx+1]]+=1

# # 写入文件
# if os.path.exists("bigram1.pkl"):
# 	os.remove("bigram1.pkl")
# cPickle.dump(bigram1,open("bigram1.pkl",'w')) 

# # 写入文件
# if os.path.exists("bigram12.pkl"):
# 	os.remove("bigram12.pkl")
# cPickle.dump(bigram12,open("bigram12.pkl",'w')) 



bigram1 = cPickle.load(open("bigram1.pkl","r"))
bigram12 = cPickle.load(open("bigram12.pkl","r"))

x = np.array([1,2,3,4,5,0,0])
y_pred_softmax = np.random.rand(5,7)

def gener_most_posible(x,y_pred_softmax,Ncand=5,Nprob=10):
	y_pred_cand = np.zeros((Ncand,7),dtype = 'int')
	# 每个位置选出最可能的Ncand个字
	for t,char_softmax in enumerate(y_pred_softmax):
		char_cand = np.argsort(-char_softmax)[0:Ncand]
		y_pred_cand[:,t] = char_cand

	if x[-1] ==0 and x[-2] ==0:
		y_pred_cand = y_pred_cand[:,0:5]
	candidates=[]
	if y_pred_cand.shape[1] == 7:
		for idx0 in y_pred_cand[:,0]:
			for idx1 in y_pred_cand[:,1]:
				for idx2 in y_pred_cand[:,2]:
					for idx3 in y_pred_cand[:,3]:
						for idx4 in y_pred_cand[:,4]:
							for idx5 in y_pred_cand[:,5]:
								for idx6 in y_pred_cand[:,6]:
									candidates.append([start_token]+[index_to_word[idx0],
													index_to_word[idx1],
													index_to_word[idx2],
													index_to_word[idx3],
													index_to_word[idx4],
													index_to_word[idx5],
													index_to_word[idx6]]+[end_token])
	else:
		for idx0 in y_pred_cand[:,0]:
			for idx1 in y_pred_cand[:,1]:
				for idx2 in y_pred_cand[:,2]:
					for idx3 in y_pred_cand[:,3]:
						for idx4 in y_pred_cand[:,4]:
							candidates.append([start_token]+[index_to_word[idx0],
											index_to_word[idx1],
											index_to_word[idx2],
											index_to_word[idx3],
											index_to_word[idx4]]+[end_token])

	proba = np.zeros(len(candidates))
	for candi_num,candidate in enumerate(candidates):
		p = 1
		for idx in range(len(candidate)-1):
			char1 = candidate[idx]
			char2 = candidate[idx+1]
			if bigram1.has_key(char1):
				p1 = bigram1[char1]
			else:
				p1 = 0
			if bigram12.has_key(char1+','+char2):
				p12 = bigram12[char1+','+char2]
			else:
				p12 = 0
			p21 = np.log(p12) - np.log(p1)
			p+=p21
		proba[candi_num] = p

	most_prob_result = []
	most_prob_proba = []
	for i in range(Nprob):
		most_prob_result.append(candidates[np.argsort(-proba)[i]])
		most_prob_proba.append(-np.sort(-proba)[i])
	return most_prob_result,most_prob_proba
