import urllib2
from bs4 import BeautifulSoup
import re
import operator
from random import randint, uniform
from time import sleep
import sys

def random_rest():
    seconds = uniform(2,7)
    sleep(seconds)

def log_err(msg):
    sys.stderr.write(msg + '\n')


class MSAuthorExtractor(object):
    def __init__(self):
        super(MSAuthorExtractor, self).__init__()
        self.topDomainIDs = {
            'Agriculture Science' : 16,
            'Arts & Humanities' : 3,
            'Biology' : 4,
            'Chemistry' : 5,
            'Computer Science' : 2,
            'Economics & Business' : 7,
            'Engineering' : 8,
            'Environmental Sciences' : 9,
            'Geosciences' : 10,
            'Material Science' : 12,
            'Mathematics' : 15,
            'Medicine' : 6,
            'Multidisciplinary' : 1,
            'Physics' : 19,
            'Social Science' : 22,
        }

        self.base_url = 'http://academic.research.microsoft.com/RankList?entitytype=2&subDomainID=0&last=0&' #topDomainID=2&start=1&end=100
        self.url_prefix = 'http://academic.research.microsoft.com/'

    def extract(self):
        retval = {}
        try:
            for academic_field, topDomainID in self.topDomainIDs.iteritems():
                tmplist = []
                start_page = 1
                while True:
                    # 1. build current suburl
                    url = self.base_url + 'topDomainID=%s&start=%s&end=%s' % (topDomainID, start_page, start_page+99)
                    log_err(url)

                    # 2. Make connection and make soup
                    try:
                        html = urllib2.urlopen(url).read()
                    except Exception, e:
                        log_err('Caught!')
                        random_rest()
                        continue
                    soup = BeautifulSoup(html)

                    # 3. Populate items into tmp container
                    try:
                        author_list = soup.find_all('div', class_='title')
                        if len(author_list) == 0:
                            break
                    except:
                        break

                    for tmp_author in author_list:
                        try:
                            tmplist.append(tmp_author.find('h3').text)
                        except:
                            continue

                    # 4. Pagination
                    start_page += 100

                retval[academic_field] = tmplist

            self.authors = retval


        except Exception, e:
            log_err(e)


    def print_authors(self):
        for k,v in self.authors.iteritems():
            print k
            for vv in v:
                print vv.encode('utf-8').strip()


if __name__ == '__main__':
    m = MSAuthorExtractor()
    m.extract()
    m.print_authors()