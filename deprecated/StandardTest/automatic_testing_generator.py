from bs4 import BeautifulSoup
import urllib2
import re
import sys, os
from datetime import datetime
from base_recognized_site_extractor import BaseRecognizedSiteExtractor
from classifier import HMMClassifier
import pickle

class ResearchGateExtractor(BaseRecognizedSiteExtractor):
    def __init__(self):
        super(ResearchGateExtractor, self).__init__()
        self.REGEX = re.compile('https?://www\.researchgate\.net/profile/[^/]+')
        self.PREFIX = 'http://www.researchgate.net/'

    def get_paginations(self, url):
        try:
            if not self.REGEX.match(url):
                return [url]

            url = self.REGEX.findall(url)[0] + '/publications/'
            paginations = [url]

            while True:
                html = urllib2.urlopen(url).read()
                soup = BeautifulSoup(html)

                next_page = soup.find_all('a', class_='navi-next pager-link')
                if len(next_page) == 0:
                    break
                else:
                    pagination = next_page[-1].attrs['href']
                    if not pagination.startswith('http'):
                        pagination = self.PREFIX + pagination
                    url = pagination
                    paginations.append(pagination)
                
            return paginations
        except Exception, e:
            print e
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return paginations

    def locate_records(self, url):
        try:
            if not self.REGEX.match(url):  #verify if the link is valid
                return []

            records = []
            url = self.REGEX.findall(url)[0] + '/publications/'

            while True:
                html = urllib2.urlopen(url).read()
                soup = BeautifulSoup(html)

                pub_list = soup.find_all('li', class_='c-list-item li-publication')
                
                for pub in pub_list:
                    new_dict = {}

                    try:
                        new_dict['authors'] = ' '.join(pub.find('div', class_='authors').text.split())
                    except Exception, e:
                        new_dict['authors'] = ''

                    try:
                        new_dict['venue'] = pub.find('div', class_='details').text.strip()
                    except:
                        new_dict['venue'] = ''

                    try:
                        title = pub.find('span', class_='publication-title').text.strip()
                        title = title.encode("ascii", "ignore")
                        new_dict['title'] = title
                    except Exception, e:
                        new_dict['title'] = ''

                    try:
                        new_dict['year'] = ''
                        years = self.YEAR_REGEX.findall(pub.find('div', class_='details').text)
                        years = sorted([int(y) for y in years], reverse=True)
                        for year in years:
                            if year <= self.YEAR_UPPERBOUND and year >= 1980:
                                new_dict['year'] = year
                                break
                    except:
                        new_dict['year'] = ''

                    try:
                        if len(new_dict['venue']) > 0:
                            new_dict['record'] = new_dict['title'] + '. ' + new_dict['authors'] + ', ' + new_dict['venue']
                        else:
                            new_dict['record'] = new_dict['title'] + '. ' + new_dict['authors']
                    except Exception, e:
                        new_dict['record'] = ''

                    if len(new_dict['record']) > 0:
                        records.append(new_dict)


                # pagination
                next_page = soup.find_all('a', class_='navi-next pager-link')
                if len(next_page) == 0:
                    break
                else:
                    pagination = next_page[-1].attrs['href']
                    if not pagination.startswith('http'):
                        pagination = self.PREFIX + pagination
                    url = pagination



            for i in range(0, len(records)):
                print "%s: %s " % (i, records[i])
            return records
        except Exception, e:
            print e
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return records



class GoogleScholarExtractor(BaseRecognizedSiteExtractor):
    def __init__(self):
        super(GoogleScholarExtractor, self).__init__()
        self.REGEX = re.compile('https?://scholar\.google\.com/citations\?.*user=', re.DOTALL)
        self.PREFIX = 'http://scholar.google.com'

    def get_paginations(self, url):
        try:
            if not self.REGEX.match(url):
                return [url]

            if url.rfind('pagesize') == -1:
                url = url + '&pagesize=100'
            else:
                url = re.sub(r'pagesize=\d+','pagesize=100',url)

            paginations = [url]
            
            while True:
                html = urllib2.urlopen(url).read()
                soup = BeautifulSoup(html)

                anchor_list = soup.find_all('a', class_='cit-dark-link')
                pagination = ''
                for anchor in anchor_list:
                    if anchor.text.lower().rfind('next') != -1:
                        pagination = anchor.attrs['href']
                        if not pagination.startswith('http'):
                            pagination = self.PREFIX + pagination
                        break
                if len(pagination) == 0:
                    break
                else:
                    url = pagination
                    paginations.append(pagination)
                
            return paginations
        except Exception, e:
            print e
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return paginations

    def locate_records(self, url):
        try:
            if not self.REGEX.match(url):
                return []

            records = []

            # change pagesize to 100 in order to decrease paginations
            if url.rfind('pagesize') == -1:
                url = url + '&pagesize=100'
            else:
                url = re.sub(r'pagesize=\d+','pagesize=100',url)

            while True:
                html = urllib2.urlopen(url).read()
                soup = BeautifulSoup(html)

                tr_list = soup.find_all('tr', class_='cit-table item')
                for tr in tr_list:
                    new_dict = {}
                    try:
                        spans = tr.find('td', attrs={'id':'col-title'}).find_all('span', class_='cit-gray')
                        authors = ''
                        venue = ''
                        title = ''
                        if len(spans) == 2:
                            authors = spans[0].text.strip()
                            venue = spans[1].text.strip()
                        if len(spans) == 1: # assume author
                            authors = spans[0].text.strip()
                        title = tr.find('td', attrs={'id':'col-title'}).find('a', class_='cit-dark-large-link').text.strip()
                        authors = authors.encode("ascii", "ignore").replace('..', '')
                        venue = venue.encode('ascii', 'ignore').replace('..', '')
                        title = title.encode('ascii', 'ignore')
                        new_dict['authors'] = authors
                        new_dict['venue'] = venue
                        new_dict['title'] = title
                    except: #???? 0 spans
                        new_dict['authors'] = ''
                        new_dict['venue'] = ''
                        new_dict['title'] = ''

                    try:
                        new_dict['year'] = int(tr.find('td', attrs={'id':'col-year'}).text)
                    except:
                        new_dict['year'] = -1

                    try:
                        if new_dict['year'] is not None:
                            record = new_dict['title'] + ',' + new_dict['authors'] + ',' + new_dict['venue'] + ' ' + str(new_dict['year'])
                        else:
                            record = new_dict['title'] + ',' + new_dict['authors'] + ',' + new_dict['venue']
                        record = record.encode("ascii", "ignore")
                        new_dict['record'] = record
                    except Exception, e:
                        print e
                        new_dict['record'] = ''

                    if len(new_dict['record'])>0:
                        records.append(new_dict)

                # Handle Pagination
                anchor_list = soup.find_all('a', class_='cit-dark-link')
                pagination = ''
                for anchor in anchor_list:
                    if anchor.text.lower().rfind('next') != -1:
                        pagination = anchor.attrs['href']
                        if not pagination.startswith('http'):
                            pagination = self.PREFIX + pagination
                        break
                if len(pagination) == 0:
                    break
                else:
                    url = pagination


            for i in range(0, len(records)):
                print "%s: %s " % (i, records[i])
            return records
        except Exception, e:
            print e
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        return records


def autotest():
    pass    


if __name__ == '__main__':
    to_save = []
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=ZVxO6IIAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=yfpJ6gwAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=x3LTjz0AAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=WZ7Pk9QAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=wDd9NlIAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=V7orfKIAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=sYKhwewAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=sXYz3okAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=rLiK2VIAAAAJ&hl=en&oi=ao')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=pWZ7c4MAAAAJ&hl=en&oi=ao')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=pWe-crIAAAAJ')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=OvpsAYgAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=o-zSlZcAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=NAyHs3oAAAAJ')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=FzGjs7cAAAAJ&hl=en&oi=sra')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=bXBgJX8AAAAJ&hl=en&oi=ao')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=avfDMDoAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=9k04-eMAAAAJ')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=96QTY3cAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=4a3L2hEAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?user=434MYswAAAAJ&hl=en')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?sortby=pubdate&hl=en&user=nXBQn7gAAAAJ&view_op=list_works')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?sortby=pubdate&hl=en&user=e7VI_HcAAAAJ&view_op=list_works')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?sortby=pubdate&hl=en&user=3Ws6G2AAAAAJ&view_op=list_works')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?sortby=pubdate&hl=en&user=0VFi-vAAAAAJ&view_op=list_works')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?hl=en&user=ZLA5LZ8AAAAJ')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?hl=en&user=tRSx_N4AAAAJ')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?hl=en&user=eVE-15EAAAAJ&view_op=list_works&gmla=AJsN-F4zm31ag--LApcHVKT-0kxJLUwPO7cRZOGd8RS4xpoijpNkwkTgRcy7C26vfRcxcBbCGGwNKQoqBB99VJUWKt2MLylvcZCxt4eXZoRi7iBb27s8VblFk2eFMIQpBYBIoUhWXSNh')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?hl=en&user=Dw5voaMAAAAJ')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?hl=en&user=B_pmuNkAAAAJ')
    to_save += GoogleScholarExtractor().locate_records('http://scholar.google.com/citations?hl=en&user=2pW1g5IAAAAJ')

    to_save += ResearchGateExtractor().locate_records('https://www.researchgate.net/profile/Sharon_Lynn_Chu/publications/')
    to_save += ResearchGateExtractor().locate_records('https://www.researchgate.net/profile/Paul_McDonald2/?ev=hdr_xprf')
    to_save += ResearchGateExtractor().locate_records('https://www.researchgate.net/profile/Jennifer_Sandoval2/?ev=hdr_xprf')
    to_save += ResearchGateExtractor().locate_records('https://www.researchgate.net/profile/Jennifer_Duringer/?ev=hdr_xprf')
    to_save += ResearchGateExtractor().locate_records('https://www.researchgate.net/profile/Jatinder_Josan/?ev=hdr_xprf')
    to_save += ResearchGateExtractor().locate_records('https://www.researchgate.net/profile/Cynthia_Rohrbeck/?ev=hdr_xprf')

    fp = open('autotest.samples', 'wb')
    pickle.dump(to_save, fp, -1)
    fp.close()
