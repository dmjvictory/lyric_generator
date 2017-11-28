#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from lxml import etree
from selenium import webdriver

browser = webdriver.Chrome()
def down_lyric(url):
	browser.get(url)
	#extract frame
	browser.switch_to_frame('contentFrame')
	comments = int(browser.find_element_by_xpath('//span[@class="j-flag"]').text)
	if comments < 200:
		return
	root = etree.HTML(browser.page_source)
	lyric = [_.encode('utf8') for _ in root.xpath('//div[@id="lyric-content"]')[0].itertext()][: -1]
	lyric = '\n'.join(lyric)
	#print lyric
	return lyric


def get_song_list(pid):
	song_list = []
	url = 'http://music.163.com/playlist?id=%s' % pid
	browser.get(url)
	#extract frame
	browser.switch_to_frame('contentFrame')
	for a in browser.find_elements_by_xpath('//table/tbody/tr/td[2]//a'):	
		link = a.get_attribute('href')
		song_name = a.find_element_by_xpath('./b').get_attribute('title')
		song_list.append((song_name.encode('utf8'), link))
	return song_list


		
if __name__ == '__main__':
	song_list = get_song_list(sys.argv[1])
	#print len(song_list)
	for song, url in song_list:
		#print song, url
		lyric = down_lyric(url)
		if lyric is None:
			continue
		with open('%s.dat' % sys.argv[1], 'a')as writer:
			writer.write('--------\n%s\n' % song)
			writer.write(lyric + '\n')
		



