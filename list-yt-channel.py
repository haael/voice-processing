#!/usr/bin/python3

from lxml.html import document_fromstring
import requests
import json
from sys import argv

url = argv[1]


tn = 2
suffix = 'shorts'

r = requests.get(url + '/' + suffix)
#raise ValueError(url + '/' +  suffix)
if not r.text:
	raise ValueError("no text")
d = document_fromstring(r.text)
for script in d.xpath('//script'):
	if script.text and script.text.startswith('var ytInitialData = '):
		j = json.loads(script.text[20:-1])
#break

if 'microformat' not in j:
	raise ValueError("format error")
channel_title = j['microformat']['microformatDataRenderer']['title']
channel_description = j['microformat']['microformatDataRenderer']['description']
channel_thumbnail = j['microformat']['microformatDataRenderer']['thumbnail']['thumbnails'][0]['url']

jtr = j['contents']['twoColumnBrowseResultsRenderer']['tabs']
if tn < len(jtr):
	jtrn = jtr[tn]
else:
	raise ValueError("format error")

if 'tabRenderer' not in jtrn:
	raise ValueError("format error")
elif 'content' not in jtrn['tabRenderer']:
	raise ValueError("format error")

jc = jtrn['tabRenderer']['content']
if 'richGridRenderer' in jc:
	jt = jc['richGridRenderer']['contents']
elif 'sectionListRenderer' in jc:
	if 'itemSectionRenderer' in jc['sectionListRenderer']['contents'][0] and 'gridRenderer' in jc['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0]:
		jt = jc['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0]['gridRenderer']['items']
	elif 'lockupViewModel' in jc['sectionListRenderer']['contents'][0] or 'channelOwnerEmptyStateRenderer' in jc['sectionListRenderer']['contents'][0]:
		raise ValueError("format error")
	elif list(jc['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0].keys()) == ['messageRenderer']:
		raise ValueError("format error")
	else:
		raise ValueError(str(jc['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0].keys()))
else:
	raise ValueError(str(jc.keys()))

videos = []

for v in jt:
	if 'richItemRenderer' in v:
		if 'videoRenderer' in v['richItemRenderer']['content']:
			vr = v['richItemRenderer']['content']['videoRenderer']
		elif 'reelItemRenderer' in v['richItemRenderer']['content']:
			vr = v['richItemRenderer']['content']['reelItemRenderer']
		elif 'shortsLockupViewModel' in v['richItemRenderer']['content']:
			vr = v['richItemRenderer']['content']['shortsLockupViewModel']
		else:
			continue
	elif 'gridPlaylistRenderer' in v:
		vr = v['gridPlaylistRenderer']
	elif ('continuationItemRenderer' in v) or ('lockupViewModel' in v):
		continue
	else:
		raise ValueError(str(v.keys()) + " " + url + '/' + suffix)
	
	if 'videoId' in vr:
		vid = vr['videoId']
		type_ = 'video'
	elif 'playlistId' in vr:
		vid = vr['playlistId']
		type_ = 'playlist'
	elif 'entityId' in vr:
		eid = vr['entityId']
		if eid.startswith('shorts-shelf-item-'):
			vid = eid[len('shorts-shelf-item-'):]
			type_ = 'video'
		else:
			raise ValueError
	else:
		raise ValueError("3")
	
	#for key in vr.keys():
	#	raise ValueError(key, vr[key])
	
	if 'thumbnails' in vr['thumbnail']:
		thumbnail = vr['thumbnail']['thumbnails'][0]['url']
	elif 'sources' in vr['thumbnail']:
		thumbnail = vr['thumbnail']['sources'][0]['url']
	else:
		raise ValueError
	
	title = vr['title']['runs'][0]['text'] if 'title' in vr else vr['headline']['simpleText'] if 'headline' in vr else None
	description = vr['descriptionSnippet']['runs'][0]['text'] if 'descriptionSnippet' in vr else None
	length = vr['lengthText']['simpleText'] if 'lengthText' in vr else None
	published = vr['publishedTimeText']['simpleText'] if 'publishedTimeText' in vr else None # TODO: parse response
	video = (type_, vid, length, published, title, description, thumbnail)
	#videos.append(video)
	
	#for video in videos[:4]:
	#vid = video[1]
	#length = video[2]
	#published = video[3]
	#title = video[4]
	
	#default_color = '\x1b[0m'
	
	#if published:
	print(vid)
