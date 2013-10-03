import nltk
import enchant
from string import punctuation

d = enchant.Dict('en_US')
fp = open('tmp.lst', 'r')

tmp_list = [item.lower().strip() for item in fp.readlines()]

long_str = ' '.join(tmp_list)

splitted_list = nltk.wordpunct_tokenize(long_str)

new_list = []

for item in splitted_list:
	if item not in punctuation and len(item)>1:
		try:
			item = item.encode('ascii', 'strict')
			if len(item) >=5 and d.check(item):
				continue
			else:
				new_list.append(item)
		except Exception, e:
			continue
		

new_list = sorted(list(set(new_list)))

for i in new_list:
	print i
