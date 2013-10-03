from bs4 import BeautifulSoup
import urllib2
import re
import sys
import bs4
from feature import FeatureGenerator
from tokens import Tokens
from string import punctuation, ascii_uppercase
import operator
from random import randint, uniform
from time import sleep
import pickle
from utils import log_err
from os import listdir, getcwd
from utils import deprecated
from utils import LABEL_INT_MAP, INT_LABEL_MAP

def random_rest():
    seconds = uniform(2,7)
    sleep(seconds)

# Regular expressions needed
GOOGLE_SCHOLAR_REGEX = re.compile('https?://scholar\.google\.com/citations\?.*user=')
DATE_REGEX1 = re.compile('\d{2}/\d{4}')
DATE_REGEX2 = re.compile('\d{4}\s*Jan|\d{4}\s*Feb|\d{4}\s*Mar|\d{4}\s*Apr|\d{4}\s*May|\d{4}\s*Jun|\d{4}\s*Jul|\d{4}\s*Aug|\d{4}\s*Sep|\d{4}\s*Oct|\d{4}\s*Nov|\d{4}\s*Dec')

CORA_TRAINING_DIR = 'data/Cora/cite_label_at/'
CORA_TESTING_DIR = 'data/Cora/cite_test/'
CORA_RAW_FILE_PATH = 'data/Cora/tagged_references.txt'

# Prefix
GOOGLE_SCHOLAR_PREFIX = 'http://scholar.google.com'

def router(url):
    if GOOGLE_SCHOLAR_REGEX.match(url):
        return gscholar(url)
    else:
        sys.exit('Invalid url!')

def gscholar_parse_detail(url):
    new_dict = {}
    try:
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html)

        # publication detail div
        cit_dt_list = soup.find('div',class_='cit-dl').find_all('div', class_='cit-dt')
        for cit_dt in cit_dt_list:
            if cit_dt.text.lower() == 'authors':
                new_dict[cit_dt.text.lower()] = [author.strip() for author in cit_dt.next_sibling.text.split(',')]
            else:
                new_dict[cit_dt.text.lower()] = cit_dt.next_sibling.text
        
        # Ensure every attr exist
        if not new_dict.has_key('title'):
            new_dict['title'] = ''
        if not new_dict.has_key('authors'):
            new_dict['authors'] = ''
        if not new_dict.has_key('publication date'):
            new_dict['publication date'] = ''
        if not new_dict.has_key('description'):
            new_dict['description'] = ''
        return new_dict

    except Exception, e:
        return {}

def gscholar(url):
    # change pagesize to 100 in order to decrease paginations
    if url.rfind('pagesize') == -1:
        url = url + '&pagesize=100'
    else:
        url = re.sub(r'pagesize=\d+','pagesize=100',url)

    result_list = []
    detail_list = []
    try:
        while True:
            try:
                html = urllib2.urlopen(url).read()
            except:
                random_rest()
                sys.stderr.write('Caught!\n')
                continue

            soup = BeautifulSoup(html)

            # publication div
            tr_list = soup.find_all('tr', class_='cit-table item')
            for tr in tr_list:
                detail_link = tr.find('td',id='col-title').find('a').attrs['href']
                if not detail_link.startswith('http'):
                    detail_link = GOOGLE_SCHOLAR_PREFIX + detail_link
                detail_list.append(detail_link)

            # Handle Pagination
            anchor_list = soup.find_all('a', class_='cit-dark-link')
            pagination = ''
            for anchor in anchor_list:
                if anchor.text.lower().rfind('next') != -1:
                    pagination = anchor.attrs['href']
                    if not pagination.startswith('http'):
                        pagination = GOOGLE_SCHOLAR_PREFIX + pagination
                    break

            if len(pagination) == 0:
                break
            else:
                url = pagination

        # Retrieve details from every detail page
        for detail_link in detail_list:
            new_dict = gscholar_parse_detail(detail_link)
            if len(new_dict) > 0:
                result_list.append(new_dict)

        return result_list

    except Exception, e:
        raise

#################################################################End of Crawler

# Label sequence as a whole
# Notes:
# 1. Au, Ti, Vn, Dt
# 2. Ti, Au, Vn, Dt
# 3. Variations of authors
def get_training_samples(url):
    log_err('\tGetting Training sample')
    raw_results = router(url)
    log_err('\tData retrieved. Preprocessing...')
    observation_list = []
    label_list = []
    records = []

    feature_generator = FeatureGenerator()
    token_generator = Tokens()

    for raw_result in raw_results:
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []

        authors = raw_result['authors']
        title = raw_result['title']        
        title_copy = raw_result['title']

        try:
            venue = raw_result['conference name']
            venue_copy = raw_result['conference name']
        except:
            venue = ''
            venue_copy = ''
        try:
            venue = raw_result['journal name']
            venue_copy = raw_result['journal name']
        except:
            venue = ''
            venue_copy = ''

        if len(venue) > 0:
            try:
                volume = raw_result['volume']
            except:
                volume = ''
            try:
                issue = raw_result['issue']
            except:
                issue = ''
            try:
                page = raw_result['page']
            except:
                page = ''

            venue += ' ' + volume + ' ' + issue + ' ' + page
            venue_copy += ' ' + volume + ' ' + issue + ' ' + page


        date = raw_result['publication date'][:4]

        # FN: 0
        # LN: 1
        # DL: 2
        # TI: 3
        # VN: 4
        # DT: 5

        # Author -> Title -> ...
        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author += ' , '
            tmp_record += author
            tmp_label_list += [0] * (feature_generator.token_length(author)-2)
            tmp_label_list += [1,2]
                
        # title
        title += ' , '
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)
        tmp_label_list += [2]

        # venue
        if len(venue) > 0:
            venue += ' , '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)

        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Title -> Author -> ...
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        # title
        # title += ' , '
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author += ' , '
            tmp_record += author
            tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
            tmp_label_list += [1,2]
                
        # venue
        if len(venue) > 0:
            # venue += ' , '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))



        #=================================Variations of authors=================================
        # Changing order, inserting dot, and probably insert comma as delimiter inside of names
        # This part of variations is very sensitive to what sample source to choose from,
        # for example, Google scholar is the current source of samples, and on gscholar, 
        # most names are in format of JW Han.  <-- Prior knowledge
        # Read more Learn more Change the Globe !!!
        log_err('\tGenerating multiple cases for name variations... ')
        # ================================A. B
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        # authors
        for author in authors:
            if len(author) == 0:
                continue

            #???? BUG!!!! split() doesn't mean tokenization
            author_tokens = token_generator.tokenize(author)['tokens']  # Split the author in order tokens
            if len(author_tokens) == 1:     # Cannot change order or anything, so leave this name alone, and pass to the next name
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Insert dot
                author = author_tokens[0] + '.' + author_tokens[1] + ' , '  # A. B
                tmp_token_length = token_generator.token_length(author)
                tmp_record += author
                tmp_label_list += [0]*(tmp_token_length-2) + [1,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # title
        # title += ' , '
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # venue
        if len(venue) > 0:
            # venue += ' , '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # Title -> Author -> ...
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']  # Split the author in order to
            if len(author_tokens) == 1:     # Cannot change order or anything, so leave this name alone, and pass to the next name
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Insert dot
                author = author_tokens[0] + '.' + author_tokens[1] + ' , '  # A. B
                tmp_token_length = token_generator.token_length(author)
                tmp_record += author
                tmp_label_list += [0]*(tmp_token_length-2) + [1,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # venue
        if len(venue) > 0:
            # venue += ' , '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # ================================B, 
        # authors
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Only keep lastname
                author = author_tokens[1] + ' , '  # B
                tmp_record += author
                tmp_label_list += [1,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # title
        # title += ' , '
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # venue
        if len(venue) > 0:
            # venue += ' , '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # Title -> Author -> ...
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Only keep lastname
                author = author_tokens[1] + ' , '  # B
                tmp_record += author
                tmp_label_list += [1,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # venue
        if len(venue) > 0:
            # venue += ' , '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))



        # ================================B A., 
        # authors
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Change order and insert dot
                author = author_tokens[1] + ' ' + author_tokens[0] + '.,'  # B A.,
                tmp_record += author
                tmp_label_list += [1,0,0,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # title
        # title += ' , '
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # venue
        if len(venue) > 0:
            # venue += ' , '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # Title -> Author -> ...
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Change order and insert dot
                author = author_tokens[1] + ' ' + author_tokens[0] + '.,'  # B A.,
                tmp_record += author
                tmp_label_list += [1,0,0,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # venue
        if len(venue) > 0:
            # venue += ' , '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # ================================B A.
        # authors
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Change order and insert dot
                author = author_tokens[1] + ' ' + author_tokens[0] + '. '  # B A.
                tmp_record += author
                tmp_label_list += [1,0,0]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # title
        # title += ' , '
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # venue
        if len(venue) > 0:
            # venue += ' , '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # Title -> Author -> ...
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Change order and insert dot
                author = author_tokens[1] + ' ' + author_tokens[0] + '. '  # B A.
                tmp_record += author
                tmp_label_list += [1,0,0]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # venue
        if len(venue) > 0:
            # venue += ' , '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))






        #============================================================================================================================================
        #============================================================================================================================================
        #============================================================================================================================================
        #============================================================================================================================================
        #============================================================================================================================================
        #============================================================================================================================================
        #============================================================================================================================================
        #============================================================================================================================================
        #============================================================================================================================================
        # Period Case!!!
        log_err('\tGenerating multiple cases for period as DL... ')
        # Author -> Title -> ...
        # authors
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        for author in authors:
            if len(author) == 0:
                continue
            author += ' , '
            tmp_record += author
            tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
            tmp_label_list += [1,2]
                
        # title
        title = title_copy + ' . '
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # venue
        if len(venue) > 0:
            venue = venue_copy + ' . '
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))



        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Title -> Author -> ...
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author += ' , '
            tmp_record += author
            tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
            tmp_label_list += [1,2]
                
        # venue
        if len(venue) > 0:
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # ================================A. B
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        # authors
        for author in authors:
            if len(author) == 0:
                continue

            author_tokens = token_generator.tokenize(author)['tokens']  # Split the author in order tokens
            if len(author_tokens) == 1:     # Cannot change order or anything, so leave this name alone, and pass to the next name
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Insert dot
                author = author_tokens[0] + '.' + author_tokens[1] + ' , '  # A. B
                tmp_token_length = token_generator.token_length(author)
                tmp_record += author
                tmp_label_list += [0]*(tmp_token_length-2) + [1,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # title
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # venue
        if len(venue) > 0:
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # Title -> Author -> ...
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']  # Split the author in order to
            if len(author_tokens) == 1:     # Cannot change order or anything, so leave this name alone, and pass to the next name
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Insert dot
                author = author_tokens[0] + '.' + author_tokens[1] + ' , '  # A. B
                tmp_token_length = token_generator.token_length(author)
                tmp_record += author
                tmp_label_list += [0]*(tmp_token_length-2) + [1,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # venue
        if len(venue) > 0:
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # ================================B, 
        # authors
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Only keep lastname
                author = author_tokens[1] + ' , '  # B
                tmp_record += author
                tmp_label_list += [1,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # title
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # venue
        if len(venue) > 0:
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # Title -> Author -> ...
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Only keep lastname
                author = author_tokens[1] + ' , '  # B
                tmp_record += author
                tmp_label_list += [1,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # venue
        if len(venue) > 0:
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))



        # ================================B A., 
        # authors
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Change order and insert dot
                author = author_tokens[1] + ' ' + author_tokens[0] + '.,'  # B A.,
                tmp_record += author
                tmp_label_list += [1,0,0,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # title
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # venue
        if len(venue) > 0:
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # Title -> Author -> ...
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Change order and insert dot
                author = author_tokens[1] + ' ' + author_tokens[0] + '.,'  # B A.,
                tmp_record += author
                tmp_label_list += [1,0,0,2]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # venue
        if len(venue) > 0:
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

       
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # ================================B A.
        # authors
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Change order and insert dot
                author = author_tokens[1] + ' ' + author_tokens[0] + '. '  # B A.
                tmp_record += author
                tmp_label_list += [1,0,0]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # title
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # venue
        if len(venue) > 0:
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))


        # Title -> Author -> ...
        tmp_record = ''
        tmp_observation_list = []
        tmp_label_list = []
        tmp_record += title
        tmp_label_list += [3] * (feature_generator.token_length(title)-1)    #!!!!
        tmp_label_list += [2]

        # authors
        for author in authors:
            if len(author) == 0:
                continue
            author_tokens = token_generator.tokenize(author)['tokens']
            if len(author_tokens) == 1:     
                author += ' , '
                tmp_record += author
                tmp_label_list += [1,2]
            elif len(author_tokens) == 2:   # Change order and insert dot
                author = author_tokens[1] + ' ' + author_tokens[0] + '. '  # B A.
                tmp_record += author
                tmp_label_list += [1,0,0]
            else:                           # name contains more than two tokens, just leave it for now
                author += ' , '
                tmp_record += author
                tmp_label_list += [0] * (feature_generator.token_length(author)-2)    #!!!!
                tmp_label_list += [1,2]
                
        # venue
        if len(venue) > 0:
            tmp_record += venue
            tmp_label_list += [4] * (feature_generator.token_length(venue)-1)    #!!!!
            tmp_label_list += [2]

        # date
        if len(date) > 0:
            tmp_record += date
            tmp_label_list += [5] * feature_generator.token_length(date)    #!!!!

        
        # Aggregate and append
        label_list.append(tmp_label_list)
        records.append(tmp_record)
        observation_list.append(feature_generator.build(tmp_record))





    # =============================================================================Verbose: Print the training set
    for record, observation, label in zip(records, observation_list, label_list):
        for rr, oo, ll in zip(token_generator.tokenize(record)['tokens'], observation, label):
            if ll == 0:
                ll = 'FN'
            elif ll == 1:
                ll = 'LN'
            elif ll == 2:
                ll = 'DL'
            elif ll == 3:
                ll = 'TI'
            elif ll == 4:
                ll = 'VN'
            elif ll == 5:
                ll = 'DT'
            print oo, '\t', ll.encode('utf-8'), '\t', rr.encode('utf-8')
        print '\n\n'

    return observation_list, label_list




# Generating training samples from raw Cora dataset
# All together 13 labels: 
#   institution
#   tech
#   title
#   note
#   author  ????fn ln?
#   location
#   booktitle
#   editor
#   date
#   pages
#   journal
#   publisher
#   volume
def get_training_samples_raw_cora():
    fp = open(CORA_RAW_FILE_PATH, 'r')
        
    feature_generator = FeatureGenerator()

    observation_list = []
    label_list = []

    for data in fp:
        # Parse every piece of training data(single piece of tagged publication)
        soup = BeautifulSoup(data)
        tmp_observation_list = []
        tmp_label_list = []
        for child in soup.body.children:
            if type(child) is bs4.element.Tag:
                raw_label = child.name
                raw_text = child.text

                label = LABEL_INT_MAP[raw_label]    # int label
                feature_vectors = feature_generator.build(raw_text)
                feature_generator.print_features()
                tmp_observation_list += feature_vectors
                tmp_label_list += [label] * len(feature_vectors)
            else:
                continue

        observation_list.append(tmp_observation_list)
        label_list.append(tmp_label_list)


    feature_generator.close_connection()
    return observation_list, label_list



if __name__ == '__main__':
    # t1 = get_training_samples2('http://scholar.google.com/citations?user=WZ7Pk9QAAAAJ&hl=en')
    # t2 = get_training_samples2('http://scholar.google.com/citations?hl=en&user=bAa___kAAAAJ&pagesize=100&view_op=list_works')
    # t3 = get_training_samples2('http://scholar.google.com/citations?hl=en&user=Kv9AbjMAAAAJ&pagesize=100&view_op=list_works')
    # t4 = get_training_samples2('http://scholar.google.com/citations?hl=en&user=WHOHV3AAAAAJ&pagesize=100&view_op=list_works')
    get_training_samples_raw_cora()

    # get_test_samples('http://scholar.google.com/citations?user=x3LTjz0AAAAJ&hl=en')
    # get_test_samples('http://scholar.google.com/citations?user=WZ7Pk9QAAAAJ&hl=en')