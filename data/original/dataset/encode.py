labels = open('Genre_annotations_sheet.txt')
labels.readline()
label_dict={}
for line in labels:
	did,primary,secondary,tertiary,hard=line[:-1].split('\t')
	hard = hard=='H'
	label_dict[did]={'primary':primary,'secondary':secondary,'tertiary':tertiary,'hard':hard}
import re
id_re=re.compile(r' id="(.+?)"')
url_re=re.compile(r' url="(.+?)"')
crawled_re=re.compile(r' crawled="(.+?)"')
def text_by_text(stream):
	text={}
	for line in stream:
		if line.startswith('<text '):
			text['id']=id_re.search(line).group(1)
			text['url']=url_re.search(line).group(1)
			text['crawled']=crawled_re.search(line).group(1)
			text.update(label_dict[text['id']])
			text['paragraphs']=[]
		elif line.startswith('<p>'):
			duplicate=False
			keep=True
		elif line.startswith('<p '):
			duplicate=True
			keep = 'leave' in line
		elif line=='</text>\n':
			yield text
			text={}
		elif line.startswith('</p>'):
			continue
		elif not line.strip()=='':
			text['paragraphs'].append({'text':line.strip(),'duplicate':duplicate,'keep':keep})
texts=list(text_by_text(open('Genre_annotated_corpus.txt')))
import json
print(len(texts))
json.dump(texts,open('dataset.json','w'),indent=4)
