from string import punctuation, ascii_uppercase
from datetime import datetime
import re
import nltk

# ???? may extend to more functionalities
class Tokens(object):
    """Token: Take a string(one line, one record) and split it into token segments"""
    def __init__(self, record):
        super(Tokens, self).__init__()
        self.record = record
        self.tokens = None
        self.tokenize()

    # With some preprocessing
    def tokenize(self):
        tmp_tokens = nltk.wordpunct_tokenize(self.record.replace(';', ' ; ').replace(',', ' , ').replace('(', ' ( '))
        # tmp_tokens = self.record.replace(',', ' ').split()   # split by space and comma

        # Remove commas in quotations, [Quotation constraint]
        quotation_format = ['\"',  '.\"',  '\".',  ',\"',  '\",',  '\"(',  ')\"',  ]
        last_quotation = []     # Record the position of closing quotation, so that we can insert DL after that
        quotation_latch = False
        for i in range(len(tmp_tokens)):
            if tmp_tokens[i] in quotation_format and (quotation_latch is False):
                quotation_latch = True
                continue
            if tmp_tokens[i] in quotation_format and (quotation_latch is True):
                quotation_latch = False
                last_quotation.append(i)
                continue
            if quotation_latch is True and (tmp_tokens[i] == ',' or tmp_tokens[i] == '.'):  # Add constraint to period also
                tmp_tokens[i] = ''

        # Replace some of the comma into ;, in order to not be eliminated by the constraint
        for i in range(len(tmp_tokens)):
            if (i+1) < len(tmp_tokens):
                token = tmp_tokens[i]
                next_token = tmp_tokens[i+1]
                if token == ',' and next_token.islower():
                    tmp_tokens[i] = ';'    

        self.tokens = []
        counter = 0
        for token in tmp_tokens:
            if len(token) > 0:
                self.tokens.append(token)
            if counter in last_quotation:
                self.tokens.append(',')     
                # self.tokens.append(';')
            counter += 1
