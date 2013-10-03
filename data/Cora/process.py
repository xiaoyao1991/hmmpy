from bs4 import BeautifulSoup

fp = open('tagged_references.txt','r')

for line in fp:
	soup = BeautifulSoup(line)
	
	author_field = soup.find_all('author')
	author_text = ''
	if len(author_field) > 0:
		author_text = author_field[0].text

	title_field = soup.find_all('title')
	title_text = ''
	if len(title_field) > 0:
		title_text = title_field[0].text

	institution_field = soup.find_all('institution')
	institution_text = ''
	if len(institution_field) > 0:
		institution_text = institution_field[0].text

	tech_field = soup.find_all('tech')
	tech_text = ''
	if len(tech_field) > 0:
		tech_text = tech_field[0].text

	note_field = soup.find_all('note')
	note_text = ''
	if len(note_field) > 0:
		note_text = note_field[0].text

	location_field = soup.find_all('location')
	location_text = ''
	if len(location_field) > 0:
		location_text = location_field[0].text

	booktitle_field = soup.find_all('booktitle')
	booktitle_text = ''
	if len(booktitle_field) > 0:
		booktitle_text = booktitle_field[0].text

	editor_field = soup.find_all('editor')
	editor_text = ''
	if len(editor_field) > 0:
		editor_text = editor_field[0].text

	date_field = soup.find_all('date')
	date_text = ''
	if len(date_field) > 0:
		date_text = date_field[0].text

	pages_field = soup.find_all('pages')
	pages_text = ''
	if len(pages_field) > 0:
		pages_text = pages_field[0].text

	journal_field = soup.find_all('journal')
	journal_text = ''
	if len(journal_field) > 0:
		journal_text = journal_field[0].text

	publisher_field = soup.find_all('publisher')
	publisher_text = ''
	if len(publisher_field) > 0:
		publisher_text = publisher_field[0].text

	volume_field = soup.find_all('volume')
	volume_text = ''
	if len(volume_field) > 0:
		volume_text = volume_field[0].text


