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


class MSVenueExtractor(object):
    def __init__(self):
        super(MSVenueExtractor, self).__init__()
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

        self.entitytypes = {
            'conference' : 3,
            'journal' : 4,
        }

        self.base_url = 'http://academic.research.microsoft.com/RankList?subDomainID=0&last=0&' #entitytype=3&topDomainID=2&start=1&end=100
        self.url_prefix = 'http://academic.research.microsoft.com/'

    def extract(self):
        retval = {}
        try:
            for academic_field, topDomainID in self.topDomainIDs.iteritems():
                tmpdict = {}
                for venue_type, entitytype in self.entitytypes.iteritems():
                    tmplist = []
                    start_page = 1
                    while True:
                        # 1. build current suburl
                        url = self.base_url + 'entitytype=%s&topDomainID=%s&start=%s&end=%s' % (entitytype, topDomainID, start_page, start_page+99)
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
                            venue_lists = soup.find_all('td', class_='rank-content')
                            if len(venue_lists) == 0:
                                break
                        except:
                            break

                        for tmp_venue in venue_lists:
                            tmplist.append(tmp_venue.text)

                        # 4. Pagination
                        start_page += 100

                    tmpdict[venue_type] = tmplist
                retval[academic_field] = tmpdict

            self.venues = retval


        except Exception, e:
            log_err(e)


    def print_venues(self):
        for k,v in self.venues.iteritems():
            for kk, vv in v.iteritems():
                print k, '\t', kk
                for v_item in vv:
                    print v_item.encode('utf-8').strip()


if __name__ == '__main__':
    m = MSVenueExtractor()
    m.extract()
    m.print_venues()