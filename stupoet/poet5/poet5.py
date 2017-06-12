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
		if i==40:break
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
Y_train = np.zeros( (X_train.shape[0],X_train.shape[1],vocab_size),dtype=np.bool )
for i,poem in enumerate(Y_train_list):
	for t,idx in enumerate(poem):
		Y_train[i , t , idx]=1

Y_test = np.zeros( (X_test.shape[0],X_test.shape[1],vocab_size),dtype=np.bool )
for i,poem in enumerate(Y_test_list):
	for t,idx in enumerate(poem):
		Y_test[i , t , idx]=1

print 'shape of X_train:',X_train.shape , '[samples , timesteps]'
print 'shape of Y_train:',Y_train.shape , '[samples , timesteps , vocab_size]'
print 'shape of X_test:',X_test.shape , '[samples , timesteps]'
print 'shape of Y_test:',Y_test.shape , '[samples , timesteps , vocab_size]'

# modeling
poems = layers.Input(shape=(X_train.shape[1],))     #(M,X_train.shape[1])  X_train.shape[1] = poem_maxlen-1
encoded_poem = Embedding(vocab_size,EMBEDDING_units)(poems)  #(M,X_train.shape[1],EMBEDDING_units)
lstm1 = recurrent.LSTM(LSTM1_cells,return_sequences=False,implementation = 2)
lstm1_output = layers.Bidirectional(lstm1,'concat')(encoded_poem)  #(M,LSTM1_cells*2)
lstm1_output = RepeatVector(X_train.shape[1])(lstm1_output)        #(M,X_train.shape[1],LSTM1_cells*2)
merged = layers.Concatenate(axis=2)([lstm1_output,encoded_poem])        ##(M,X_train.shape[1],LSTM1_cells*2+EMBEDDING_units )
lstm2 = recurrent.LSTM(LSTM2_cells,return_sequences=True,implementation = 2)
lstm2_output = layers.Bidirectional(lstm2,'concat')(merged) #(M,X_train.shape[1],LSTM2_cells*2)
# preds = layers.TimeDistributed(Dense(vocab_size, activation='softmax'))(lstm2_output)  #(M,X_train.shape[1],vocab_size)
preds = Dense(vocab_size, activation='softmax')(lstm2_output)  #(M,X_train.shape[1],vocab_size)

model = Model(poems,preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
             )
## 权值个数计算: 
print 'weights num', vocab_size*EMBEDDING_units + LSTM1_cells*4*(LSTM1_cells+EMBEDDING_units+1)*2 \
		+ LSTM2_cells*4*(LSTM2_cells+(LSTM1_cells*2+EMBEDDING_units)+1)*2 + 2*LSTM2_cells*vocab_size

#训练
print('Training')
def sample(preds, temperature=0.3):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

lastsave = -1  #最后一个完整的cycle -1表示没有训练过
import cPickle
for cycle in range(0,7):
	if cycle<lastsave:
		continue
	elif cycle==lastsave:
		print 'loading model from cycle{0}'.format(lastsave)
		#加载模型数据和weights  
		from keras.models import model_from_json
		# model = model_from_json(open('cycle{0}/my_model_architecture.json'.format(lastsave)).read())    
		# model.load_weights('cycle{0}/my_model_weights.h5'.format(lastsave))
		model = model_from_json(open('cycle{0}/my_model_architecture.json'.format(lastsave)).read())
		weights = cPickle.load(open("cycle{0}/weights.pkl".format(cycle),"rb"))
		model.set_weights(weights)
		model.compile(optimizer='adam',
              loss='categorical_crossentropy',
             ) 
		# a = model.get_config()  #可打印模型结构
	else:
		os.mkdir('cycle{0}'.format(cycle))
		model.fit(X_train, Y_train,
		          batch_size=2048,
		          epochs=10,
		          validation_split=0.05)

		# 模型存储
		json_string = model.to_json()  #等价于 json_string = model.get_config()  
		open('cycle{0}/my_model_architecture.json'.format(cycle),'w').write(json_string)    
		weights = model.get_weights()
		cPickle.dump(weights,open("cycle{0}/weights.pkl".format(cycle),"wb")) 
		# model.save_weights('cycle{0}/my_model_weights.h5'.format(cycle))    
		# 预测测试

		if os.path.exists("cycle{0}/7char_pair.txt".format(cycle)):
			os.remove("cycle{0}/7char_pair.txt".format(cycle))


		result = codecs.open("cycle{0}/7char_pair.txt".format(cycle),"a","utf-8")  
		Y_pred = model.predict(X_test)     # (M,X_train.shape[1],vocab_size)
		for i,x in enumerate(X_test):
			# print '\n============\n',
			result.write('\n============\n')

			for j,word_idx in enumerate(x):
				# print index_to_word[word_idx],
				result.write( index_to_word[word_idx] +u' ')
			# print '\n',
			result.write('\n')
			y_test = Y_test_list[i]
			for j,word_idx in enumerate(y_test):
				# print index_to_word[word_idx],
				result.write( index_to_word[word_idx] +u' ')

			# print '\n',
			result.write('\n')
			y_pred_softmax = Y_pred[i]   #(X_train.shape[1],vocab_size)
			for t,char_softmax in enumerate(y_pred_softmax):
				# char_pred = sample(char_softmax, temperature=0.3)
				# char_pred = np.argmax(char_softmax)
				# print index_to_word[int(char_pred)],
				result.write( index_to_word[int(char_pred)]+u' ')
		result.close()