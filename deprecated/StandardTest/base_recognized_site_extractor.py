from bs4 import BeautifulSoup
import urllib2
import re
import sys, os
from datetime import datetime

class BaseRecognizedSiteExtractor(object):
    """docstring for BaseRecognizedSiteExtractor"""
    def __init__(self):
        super(BaseRecognizedSiteExtractor, self).__init__()
        self.REGEX = None
        self.PREFIX = None
        self.YEAR_REGEX = re.compile('(?<=\,|\.|\s|\(|\[|\;|\/)[1|2]\d{3}(?=\,|\.|\s|\)|\[|\;|\/)')
        self.YEAR_UPPERBOUND = datetime.now().year

    def locate_records(self, url):
        pass

    def get_paginations(self, url):
        pass
