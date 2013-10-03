import enchant, sys

d = enchant.Dict('en_US')

# fp = open('raw_venues.txt', 'r')

# processed = []

# for line in fp:
# 	tmp_list = line.strip().lower().split('-')
# 	# for item in tmp_list:
# 	# 	if len(item) == 1:
# 	# 		continue
# 	# 	else:
# 	# 		processed.append(item)
# 	if len(tmp_list) > 1:
# 		for item in tmp_list[0].split():
# 			if d.check(item):
# 				continue
# 			if len(item) > 1:
# 				processed.append(item)

# processed = list(set(processed))
# processed = sorted(processed)

# for item in processed:
# 	if d.check(item):
# 		sys.stderr.write(item+'\n')
# 	print item

fp = open('countries.lst', 'r')
processed = []

for line in fp:
	processed.append(line.strip().lower())

processed = list(set(processed))
processed = sorted(processed)

for item in processed:
	print item