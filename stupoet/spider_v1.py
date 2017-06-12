#coding=utf-8
import urllib
import urllib2
import re
 
for page in range(1,500):
	url = 'http://so.gushiwen.org/type.aspx?p={0}&c=%E5%94%90%E4%BB%A3'.format(page) # 唐
	user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
	headers = { 'User-Agent' : user_agent }
	request = urllib2.Request(url,headers = headers)
	response = urllib2.urlopen(request)

	content = response.read().decode('utf-8')


	pattern = re.compile('<div class="sons".*?alt="(.*?)" />.*?' + 
		'<p><a .*?font-size:14px.*?_blank">(.*?)</a>.*?' + 
		'<p style="margin-bottom:0px;">(.*?)....<a .*?>.*?' + 
		'</div>',re.S)
	items = re.findall(pattern,content)
	print 'page:{0} , length of poems:{1}'.format(page,len(items))

	import codecs  
	file=codecs.open("TangDai.txt","a","utf-8")  

	for item in items:
		author = item[0]
		title =  item[1]
		content = item[2]
		content = content.replace(u'《','')
		content = content.replace(u'》','')
		content = content.replace(u'“','')
		content = content.replace(u'”','')
		content = content.replace(u'\n','')
		content = content.replace(u'\t','')
		content = content.replace(u' ','')

		#删掉完整注释
		pattern = re.compile(u'\\(.*?\\)|\\(.*?\\）|\\（.*?\\)|\\（.*?\\）')
		annotations = re.findall(pattern,content)
		if annotations:
			# 如果有注释
			for annotation in annotations:
				content = content.replace(annotation,'')
		#删掉一半的注释
		pattern = re.compile(u'\\(.+|\\（.+')
		annotations = re.findall(pattern,content)
		if annotations:
			# 如果有一半的注释
			for annotation in annotations:
				content = content.replace(annotation,'')

		#删掉最后一个标点后的不完整的句子（待定）
		pattern = re.compile(u'[^,.?!:;，。？！；：]+$')
		annotations = re.findall(pattern,content)
		if annotations:
			# 如果有一半的注释
			for annotation in annotations:
				content = content.replace(annotation,'')


		# print author
		# print title
		# print content
		# print '-------'

		file.write(author+'|'+title+'|'+content+'\n')
file.close()
	
