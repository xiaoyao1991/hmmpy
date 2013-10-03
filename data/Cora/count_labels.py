fp = open('labels.txt', 'r')

mapping = {}

for line in fp:
	key = line.strip()
	if mapping.has_key(key):
		mapping[key] += 1
	else:
		mapping[key] = 1

print mapping, len(mapping)