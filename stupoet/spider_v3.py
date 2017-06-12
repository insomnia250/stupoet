#coding=utf-8
import urllib
import urllib2
import re
import os
import codecs
#1109 1703 1849 1979

url = 'http://www.guoxuedashi.com/a/7176q/'
user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
headers = { 'User-Agent' : user_agent }
request = urllib2.Request(url,headers = headers)
response = urllib2.urlopen(request)

content = response.read().decode('utf-8')

pattern = re.compile(u'<dd><a href="(.*?)" target="_blank">卷.*?</a></dd>',re.S)
links = re.findall(pattern,content)

#----------------------------------------
if os.path.exists("./data/QuanSongShi/QuanSongShi.txt"):
	os.remove("./data/QuanSongShi/QuanSongShi.txt")

file=codecs.open("./data/QuanSongShi/QuanSongShi.txt","a","utf-8")  
for volume,link in enumerate(links):
	# if volume <= 255:continue
	# if volume == 20:break
	print 'volume:',volume
	url = 'http://www.guoxuedashi.com'+ link
	user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
	headers = { 'User-Agent' : user_agent }
	request = urllib2.Request(url,headers = headers)
	response = urllib2.urlopen(request)

	content = response.read().decode('utf-8')


	pattern = re.compile(u'<div class="info_txt clearfix" id="infozj_txt">.*?<div class="info_cate clearfix">',re.S)
	items = re.findall(pattern,content)

	if items:
		content = items[0]

		#删掉完整注释和HTML标签
		pattern = re.compile(u'\\(.*?\\)|\\(.*?\\）|\\（.*?\\)|\\（.*?\\）|\\【.*?\\】|<.*?>')
		annotations = re.findall(pattern,content)
		if annotations:
			# 如果有注释
			for annotation in annotations:
				content = content.replace(annotation,'')

		#删掉{}注释用〇(ling)代替
		pattern = re.compile(u'\\{.*?\\}')
		annotations = re.findall(pattern,content)
		if annotations:
			# 如果有注释
			for annotation in annotations:
				content = content.replace(annotation,u'〇')
		content = content.replace('    ','')
		content = content.replace('\r\n','|')
		content = content.replace('||','#')
		content = content.replace('|','#')
		content = content.split(u'　')
		poems = content[1:]

		for i,poem in enumerate(poems):
			#处理作者，题目格式
			title = poem.split(u'#')[0]
			titlelen = len(title)
			content = ''.join(poem.split(u'#')[1:-1])
			author = ''
			if content:
				file.write(author+'|'+title+'|'+content+'\n')

file.close()

