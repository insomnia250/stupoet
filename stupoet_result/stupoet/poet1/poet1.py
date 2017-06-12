#coding=utf-8
from functools import reduce
import re
import numpy as np
import codecs
import os
# from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent,Dropout,Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def tokenize(file,start_token,end_token):
	parsed_poem = []
	charNum = 0
	for i,line in enumerate(file):
		# if i==32:break
		content = line.strip('\n').split('|')[-1]
		poem = [start_token] + list(content.decode('utf8')) + [end_token]  #按utf8分词
		charNum += len(poem)
		parsed_poem.append(poem)
	return parsed_poem,charNum

def vectorize_stories(parsed_poem, word_to_index, poem_maxlen):
    xs = []
    for poem in parsed_poem:
        x = [word_to_index[w] for w in poem]
        xs.append(x)
    # return xs
    return pad_sequences(xs, maxlen=poem_maxlen ,padding='post',truncating='post')

def characterize(vec_poems, index_to_word):
    xs = []
    for poem in vec_poems:
        x = [index_to_word[idx] for idx in poem]
        xs.append(x)
    return xs



unknown_token = u'unknown'
start_token = u'start'
end_token = u'end'
NA_token = u'NA'

EMBEDDING_units = 50
LSTM_cells = 50


file = open('../data/QuanTangShi/QuanTangShi.txt','rU')

#切分成单字，存在parsed_poem中[ ['君','不','见','，',...] , ['山','不','在',...] ]
parsed_poem , charNum = tokenize(file,start_token,end_token)
print 'total chars number:' , charNum
print 'total poems number:' , len(parsed_poem)

#统计词频，选最多的字
from collections import Counter
vocab_freq =Counter()
print 'counting vocab_freq...'
for poem in parsed_poem:
	cnt = Counter(poem)
	vocab_freq = vocab_freq+cnt

print 'unique char num:',len(vocab_freq)-2
vocab_freq = sorted(vocab_freq.iteritems(),key=lambda d:d[1],reverse=True)[0:4500]

# 统计词频写入文件
if os.path.exists("freq.txt"):
	os.remove("freq.txt")
result = codecs.open("freq.txt","a","utf-8") 
for char,cnt in vocab_freq:
	result.write(char+':'+str(cnt)+'\n')
result.close()



# word 和 int 映射
index_to_word = [x[0] for x in vocab_freq] 
index_to_word = [NA_token] + index_to_word
index_to_word.append(unknown_token)
if not start_token in index_to_word: index_to_word.append(start_token)
if not end_token in index_to_word: index_to_word.append(end_token)

word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
vocab_size = len(word_to_index)
print 'vocab size(with {0}(0), {1}, {2}, {3}({4}):{5}'\
		.format(NA_token, start_token, end_token,unknown_token,vocab_size-1, vocab_size)

#用u'unknown'代替低频词
for i, poem in enumerate(parsed_poem):
    parsed_poem[i] = [w if w in word_to_index else unknown_token for w in poem]

# 最大长度
# poem_maxlen = max(map(len, (x for x in parsed_poem)))
poem_maxlen = 50
print 'poem_maxlen:' , poem_maxlen


vec_poems = vectorize_stories(parsed_poem, word_to_index, poem_maxlen)
print vec_poems[0]
# Create the training data
X = np.asarray([vec[0:-1] for vec in vec_poems])
Y_list = [vec[1: ] for vec in vec_poems]

Y = np.zeros( (X.shape[0],X.shape[1],vocab_size),dtype=np.bool )
for i,poem in enumerate(Y_list):
	for t,idx in enumerate(poem):
		Y[i , t , idx]=1


print 'shape of X:',X.shape , '[samples , timesteps]'
print 'shape of Y:',Y.shape , '[samples , timesteps , vocab_size]'

# train test split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.0025, random_state=42)


# modeling
poems = layers.Input(shape=(X_train.shape[1],))     #(M,X_train.shape[1])  X_train.shape[1] = poem_maxlen-1
encoded_poem = Embedding(vocab_size,EMBEDDING_units)(poems)  #(M,X_train.shape[1],EMBEDDING_units)
encoded_poem = Dropout(0.5)(encoded_poem)
lstm_output = recurrent.LSTM(LSTM_cells,return_sequences=True,implementation = 2)(encoded_poem)   #(M,X_train.shape[1],LSTM_cells)
lstm_output = Dropout(0.5)(lstm_output)
preds = Dense(vocab_size, activation='softmax')(lstm_output)

model = Model(poems ,preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
             )


## 权值个数计算: 
print 'weights num', vocab_size*EMBEDDING_units + LSTM_cells*4*(LSTM_cells+EMBEDDING_units+1) + LSTM_cells*vocab_size

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

for cycle in range(0,10):
	os.mkdir('cycle{0}'.format(cycle))
	model.fit(X_train, Y_train,
	          batch_size=32,
	          epochs=20,
	          validation_split=0.01)

	# # 模型存储
	json_string = model.to_json()  #等价于 json_string = model.get_config()  
	open('cycle{0}/my_model_architecture.json'.format(cycle),'w').write(json_string)    
	model.save_weights('cycle{0}/my_model_weights.h5'.format(cycle))    

	# #加载模型数据和weights  
	# from keras.models import model_from_json
	# model = model_from_json(open('my_model_architecture.json').read())    
	# model.load_weights('my_model_weights.h5')    
	# # a = model.get_config()  #可打印模型结构

	# 预测测试

	if os.path.exists("cycle{0}/7char_fill.txt".format(cycle)):
		os.remove("cycle{0}/7char_fill.txt".format(cycle))


	result = codecs.open("cycle{0}/7char_fill.txt".format(cycle),"a","utf-8")  
	X_test[:,9:]=0   #start + 7chars + ','
	
	for i,x in enumerate(X_test):
		print '\n============',
		result.write('\n============')
		print '\n-------test------',
		result.write('\n-------test------')
		for j,word_idx in enumerate(x):
			print index_to_word[word_idx],
			result.write( index_to_word[word_idx] +u' ')
			if j==8 : result.write(u'||')


		print '\n-------pred------',
		result.write('\n-------pred------')
		input_vec = np.zeros( (1,X_test.shape[1]) )
		char_pred = word_to_index[start_token]
		for j,word_idx in enumerate(x):
			if word_idx!=0:    #如果输入不是空，取输入，是空的话取上一次输出值
				input_vec[0][j] = word_idx
			else:
				input_vec[0][j] = char_pred
			poem_pred_softmax = model.predict(input_vec)
			# char_pred = np.argmax(poem_pred_softmax,axis=2)[0][j]  #vectorize
			# # sample type
			char_pred = sample(poem_pred_softmax[0][j], temperature=0.3)
			print index_to_word[int(input_vec[0][j])],
			result.write( index_to_word[int(input_vec[0][j])] +u' ')
			if j==8 : result.write(u'||')
		print index_to_word[int(char_pred)],
		result.write( index_to_word[int(char_pred)]+u'\n')
	result.close()