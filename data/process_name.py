from string import punctuation

fp = open('known_name.lst', 'r')

big_list = []

for line in fp:
	big_list.append(line.lower().strip())

dedu_list = sorted(list(set(big_list)))

for name in dedu_list:
	print name