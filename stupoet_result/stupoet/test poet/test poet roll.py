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
			pair1 =  poem[pairIdx*2*delta : (pairIdx*2+1)*delta-1] +pad*['NA','NA']
			pair2 = poem[(pairIdx*2+1)*delta : (pairIdx*2+2)*delta-1]+ pad*['NA','NA']
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

# #统计词频，选最多的字
# from collections import Counter
# vocab_freq =Counter()
# print 'counting vocab_freq...'
# for i,pair in enumerate(pairs):
# 	print i
# 	cnt = Counter(pair)
# 	vocab_freq = vocab_freq+cnt

# print 'unique char num:',len(vocab_freq)-2
# vocab_freq = sorted(vocab_freq.iteritems(),key=lambda d:d[1],reverse=True)[0:5500]

# # 统计词频写入文件
# if os.path.exists("freq.txt"):
# 	os.remove("freq.txt")
# result = codecs.open("freq.txt","a","utf-8") 
# for char,cnt in vocab_freq:
# 	result.write(char+':'+str(cnt)+'\n')
# result.close()

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
index_to_word = [end_token] + index_to_word
index_to_word = [start_token] + index_to_word

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

# 最大长度
# pair_maxlen = max(map(len, (x for x in pairs)))
pair_maxlen = 14
print 'pair_maxlen:' , pair_maxlen


vec_poems = vectorize_stories(pairs, word_to_index, pair_maxlen)
print vec_poems[0]
# Create the training data
X = np.asarray([vec[0:7] for vec in vec_poems])
Y_list = np.asarray([vec[7: ] for vec in vec_poems])



# train test split
X_train, X_test, Y_train_list, Y_test_list = train_test_split( X, Y_list, test_size=0.001, random_state=42)
del X,Y_list

Y_test = np.zeros( (X_test.shape[0],X_test.shape[1],vocab_size),dtype=np.bool )
for i,poem in enumerate(Y_test_list):
	for t,idx in enumerate(poem):
		Y_test[i , t , idx]=1


# modeling
poet = 'poet6'
cycle = 6
print 'loading model from {0} cycle{1}'.format(poet,cycle)
#加载模型数据和weights  
from keras.models import model_from_json
import cPickle
# model = model_from_json(open('cycle{0}/my_model_architecture.json'.format(poet,cycle)).read())    
# model.load_weights('cycle{0}/my_model_weights.h5'.format(poet,cycle))
model = model_from_json(open('../{0}/cycle{1}/my_model_architecture.json'.format(poet,cycle)).read())
weights = cPickle.load(open("../{0}/cycle{1}/weights.pkl".format(poet,cycle),"rb"))
model.set_weights(weights)
model.compile(optimizer='adam',
      loss='categorical_crossentropy',
     ) 

# 预测测试
def poly_sample(preds, temperature=0.3):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def print_pred(result,x, sample):
	input_poem = np.zeros( (1,len(x)) )
	input_poem[0] = x
	roll_poem = np.zeros( (1,len(x)) )
	char_pred = word_to_index[start_token]
	for j,word_idx in enumerate(x):
		roll_poem[0][j] = char_pred
		poem_pred_softmax = model.predict([input_poem,roll_poem])
		if sample=='poly':
			char_pred = poly_sample(poem_pred_softmax[0][j], temperature=0.3)
		elif sample=='max':
			char_pred = np.argmax(poem_pred_softmax,axis=2)[0][j]  #vectorize
		else:
			char_pred = np.argsort(-poem_pred_softmax[0][j])[sample] # [1~len(vocab_size)] 次大值
		print  index_to_word[int(char_pred)],
	print '\n'
def write_pred(result,x, sample):
	input_poem = np.zeros( (1,len(x)) )
	input_poem[0] = x
	roll_poem = np.zeros( (1,len(x)) )
	char_pred = word_to_index[start_token]
	for j,word_idx in enumerate(x):
		roll_poem[0][j] = char_pred
		poem_pred_softmax = model.predict([input_poem,roll_poem])
		if sample=='poly':
			char_pred = poly_sample(poem_pred_softmax[0][j], temperature=0.3)
		elif sample=='max':
			char_pred = np.argmax(poem_pred_softmax,axis=2)[0][j]  #vectorize
		else:
			if j==0:
				char_pred = np.argsort(-poem_pred_softmax[0][j])[sample] # [1~len(vocab_size)] 次大值
			else:
				char_pred = np.argsort(-poem_pred_softmax[0][j])[0]
		result.write( index_to_word[int(char_pred)])
	result.write('\n')

if os.path.exists("7char_pair_{0}({1}).txt".format(poet,cycle)):
	os.remove("7char_pair_{0}({1}).txt".format(poet,cycle))


result = codecs.open("7char_pair_{0}({1}).txt".format(poet,cycle),"a","utf-8")  

for i,x in enumerate(X_test):
	# print '\n============\n',
	result.write('\n============\n')

	for j,word_idx in enumerate(x):
		# print index_to_word[word_idx],
		result.write( index_to_word[word_idx])
	# print '\n',
	result.write('\n')
	# y_test = Y_test_list[i]
	# for j,word_idx in enumerate(y_test):
	# 	# print index_to_word[word_idx],
	# 	result.write( index_to_word[word_idx])
	# print '\n',
	# result.write('\n')
	
	write_pred(result,x, sample='max')
	write_pred(result,x, sample=1)
	write_pred(result,x, sample=2)
	write_pred(result,x, sample=3)
result.close()


# input_poem = raw_input("Enter your input: ")  # 输入
# print "Received input is : ", input_poem.decode('utf8')
query = [u'春风小雨润如酥',u'桃面晚秋春好处']
input_poem = [list(poem) + (7 - len(poem))*['NA'] for poem in query]
input_poem = vectorize_stories(input_poem, word_to_index, pair_maxlen)
X_test = np.asarray([vec[0:7] for vec in input_poem])

for i,x in enumerate(X_test):
	print '============='
	for word_idx in x:
		print index_to_word[word_idx],
	print '\n',
	print_pred(result,x, sample='poly')
	print_pred(result,x, sample='poly')
	print_pred(result,x, sample='poly')
	print_pred(result,x, sample='poly')