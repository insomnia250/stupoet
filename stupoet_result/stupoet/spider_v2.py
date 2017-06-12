#coding=utf-8
import urllib
import urllib2
import re


import codecs
import os
if os.path.exists("./data/QuanTangShi/QuanTangShi.txt"):
	os.remove("./data/QuanTangShi/QuanTangShi.txt")


file=codecs.open("./data/QuanTangShi/QuanTangShi.txt","a","utf-8")  

#1109 1703 1849 1979
for page in range(1109,2011): 
	if page in [1703,1849,1979]:continue
	url = 'http://www.gushiwen.org/wen_{0}.aspx'.format(page)
	user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
	headers = { 'User-Agent' : user_agent }
	request = urllib2.Request(url,headers = headers)
	response = urllib2.urlopen(request)

	content = response.read().decode('utf-8')

	pattern = re.compile(u'<br />.*?卷.*?<p style="text-align.*?',re.S)

	items = re.findall(pattern,content)
	if items:
		content = items[0]
		#删掉完整注释和HTML标签
		pattern = re.compile(u'\\(.*?\\)|\\(.*?\\）|\\（.*?\\)|\\（.*?\\）|<.*?>')
		annotations = re.findall(pattern,content)
		if annotations:
			# 如果有注释
			for annotation in annotations:
				content = content.replace(annotation,'')

		# 嵌套的影响，得再删一次
		#删掉完整注释和HTML标签
		pattern = re.compile(u'\\(.*?\\)|\\(.*?\\）|\\（.*?\\)|\\（.*?\\）|<.*?>')
		annotations = re.findall(pattern,content)
		if annotations:
			# 如果有注释
			for annotation in annotations:
				content = content.replace(annotation,'')

		#删掉一半的注释
		pattern = re.compile(u'\\(.+|\\（.+|<.+')
		annotations = re.findall(pattern,content)
		if annotations:
			# 如果有一半的注释
			for annotation in annotations:
				content = content.replace(annotation,'')

		poems =  content.split(u'_')[1:]
		print 'page {0} , poem {1}'.format(page,len(poems))


		for i,poem in enumerate( poems):
			#处理作者，题目格式
			title = poem.split(u'」')[0].split(u'「')[1]
			author = poem.split(u'」')[-1].split(' ')[0]
			authorlen = len(author)
			authorcontent = poem.split(u'」')[-1]
			content = authorcontent[authorlen:]

			#删掉名句作者注释
			pattern = re.compile(u'——.*? ')
			annotations = re.findall(pattern,content)
			if annotations:
				# 如果有
				for annotation in annotations:
					content = content.replace(annotation,'')

			content = content.replace(u'《','')
			content = content.replace(u'》','')
			content = content.replace(u'“','')
			content = content.replace(u'”','')
			content = content.replace(u'\n','')
			content = content.replace(u'\t','')
			content = content.replace(u' ','')
			content = content.replace(u'　','')

			#删掉最后一个标点后的不完整的句子（待定）
			pattern = re.compile(u'[^,.?!:;，。？！；：]+$')
			annotations = re.findall(pattern,content)
			if annotations:
				# 如果有一半的注释
				for annotation in annotations:
					content = content.replace(annotation,'')

			file.write(author+'|'+title+'|'+content+'\n')
file.close()
