import re
import sys
import enchant
from string import punctuation, ascii_uppercase
from tokens import Tokens
import nltk
from datetime import datetime
import json

STANDARD_PIPELINE = [
                        'f_is_name_abbrev',
                        'f_is_apostrophes',
                        'f_is_capitalized',
                        'f_is_all_upper',
                        'f_is_english',
                        'f_has_both_char_and_digit',
                        'f_is_delimiter',
                        'f_is_quotation',
                        'f_is_punctuation',
                        'f_has_digit',
                        'f_is_all_digit',
                        'f_is_possible_page_number',
                        'f_is_month',
                        'f_is_possible_year',
                        'f_is_in_namelist',
                        'f_is_ordinal',
                        'f_is_in_venuelist',
                        'f_has_lbracket_before',
                        'f_has_rbracket_after',
                        'f_has_quotation_before',
                        'f_has_quotation_after',
                        'f_is_possible_volume',
                        'f_is_at_second_half_of_string',
                        'f_has_delimiter_before',
                        'f_has_delimiter_after',
                        'f_is_an_and_between_two_names',
                        'f_is_followed_by_year',
                        'f_is_possible_new_notion',
]

PARTIAL_PIPELINE = [
                        'f_is_capitalized',
                        'f_is_all_upper',
                        'f_is_english',
                        'f_has_both_char_and_digit',
                        'f_is_delimiter',
                        'f_is_punctuation',
                        'f_is_sequential_punctuation',
                        'f_has_digit',
                        'f_is_all_digit',
                        'f_is_possible_year',
                        'f_is_in_namelist',
                        'f_is_in_venuelist',
                        'f_is_fname_abbrev',
                        'f_is_preceeded_by_delimiter',
                        'f_is_followed_by_delimiter',
                        'f_is_possible_page_number',
                        'f_is_an_and_between_two_names',
                        'f_is_punctuation_in_name',
                        'f_is_followed_by_year',
                        'f_is_possible_new_notion',
]


class FeatureGenerator(object):
    """
        @param:
            record -> piece of raw_text, or a list of tokens
    """
    def __init__(self, feature_for_separate_model=False):
        super(FeatureGenerator, self).__init__()
        self.dictionary = enchant.Dict('en_US')
        self.token_generator = Tokens()     # Connection established!
        self.record = None
        self.tokens = []
        self.features = None      # list of list of features for every name; e.g. [[1,1,1,1],[...], ...]

        
        # Regex setup
        self.NUM_REGEX = re.compile('\d')
        self.CHAR_DIGIT_MIX_REGEX = re.compile('((^[a-zA-Z]+\d{4}$)|(^[a-zA-Z]+\d{2}$))|((^\d{4}[a-zA-Z]+$)|(^\d{2}[a-zA-Z]+$))', re.MULTILINE)
        self.NAME_ABBREV_REGEX = re.compile('([A-Z]\.-[A-Z]\.)|([A-Z]\.-[A-Z])|([A-Z]\.-)|(([A-Z]\.)+)|(O\'[A-Z][a-z]+)')   #C.P.; C.-C.; O'Reilly
        self.PAGE_NO_REGEX = re.compile('\d+-\d+')

        # Gazzatte setup
        self.DELIMITERS = [',', '.', ]
        self.LBRACKET = ['(', '[', '{', '<', ]
        self.RBRACKET = [')', ']', '}', '>', ]
        self.APOSTROPHES = ["'s", "'re", "'d", ]
        self.QUOTATIONS = ['"', "''", "``", ]
        self.MONTHS = ['Janurary', 'February', 'March', 'April','May','June','July','August','September','October','November','December']
        self.NAME_LIST = [item.strip() for item in open('data/name.lst','r').readlines()]
        self.VENUE_LIST = [item.strip() for item in open('data/venue.lst','r').readlines()]
        self.ORDINAL_LIST = [item.strip() for item in open('data/ordinal.lst','r').readlines()]
        # self.CITY_LIST = [item.strip() for item in open('data/cities.lst','r').readlines()]
        self.COUNTRY_LIST = [item.strip() for item in open('data/countries.lst','r').readlines()]

        if feature_for_separate_model:
            self.pipeline = PARTIAL_PIPELINE
        else:
            self.pipeline = STANDARD_PIPELINE

    def close_connection(self):
        self.token_generator.close_connection()


    def build(self, record):
        self.record = record

        features = []
        need_tokenize = True
        if type(self.record) is list:
            need_tokenize = False
        else:
            need_tokenize = True

        # record raw texts
        if need_tokenize:
            response_obj = self.token_generator.tokenize(self.record)
            self.tokens = response_obj['tokens']

        # Already tokenized input
        else:   
            self.tokens = self.record

        self.num_tokens = len(self.tokens)  # count how many tokens are there in this piece of text.

        for i in range(self.num_tokens):
            sub_features = []
            for pipe in self.pipeline:
                action = getattr(self, pipe)
                sub_features.append(action(i))
            features.append(sub_features)
        self.features = features

        return features

    def token_length(self, record):
        return self.token_generator.token_length(record)


    def print_features(self):
        for i in range(self.num_tokens):
            print self.features[i], '\t\t', self.tokens[i]


    ################################### Feature functions ###################################
    # Feature output format:
    # [
    #   [([1,0,0,1], 1), ([1,1,1,1], 0), (...)...], <-- One piece of training sample (x, y) where x=x1x2x3...xm, y=y1y2y3...ym <-- a sentence representation in feature vectors, in sequence
    #   [.......................],  <-- another sentence, parallel with the previous sentence, independent processed
    #   ...
    # ]
    # Assume segment is space-delimited, so it's a feature for the segmentm challenge will be tokenizing
    ################################### Local Features #####################################

    # C.B. or C.-C 
    def f_is_name_abbrev(self, idx):
        token = self.tokens[idx]
        if self.NAME_ABBREV_REGEX.match(token) is None:
            return 0
        return 1

    def f_is_apostrophes(self, idx):
        token = self.tokens[idx]
        return int(token in self.APOSTROPHES)

    def f_is_capitalized(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(token[0].isupper())

    def f_is_all_upper(self, idx):
        token = self.tokens[idx]
        if len(token) <= 2:
            return 0
        return int(token.isupper())

    def f_is_english(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(self.dictionary.check(token.lower()) and len(token) > 1)

    def f_has_both_char_and_digit(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        if self.CHAR_DIGIT_MIX_REGEX.search(token) is None:
            return 0
        return 1

    def f_is_delimiter(self, idx):
        token = self.tokens[idx]
        if len(token) != 1:
            return 0
        return int(token in self.DELIMITERS)

    def f_is_quotation(self, idx):
        token = self.tokens[idx]
        return int(token in self.QUOTATIONS)

    def f_is_punctuation(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(len(token) == 1 and token in punctuation)

    # def f_is_sequential_punctuation(self, idx): #e.g. K.C.-C. Chang
    #     token = self.tokens[idx]
    #     if len(token) <= 1:
    #         return 0
    #     ret = 1
    #     for t in token:
    #         if t not in punctuation:
    #             ret = 0
    #             break
    #     return ret

    def f_has_digit(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        if self.NUM_REGEX.search(token) is None:
            return 0
        return 1

    def f_is_all_digit(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(token.isdigit())

    def f_is_possible_page_number(self, idx):
        token = self.tokens[idx]
        if self.PAGE_NO_REGEX.match(token) is None:
            return 0
        return 1

    def f_is_month(self, idx):
        token = self.tokens[idx]
        return int(token in self.MONTHS)

    def f_is_possible_year(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(token.isdigit() and len(token)==4 and int(token)>= 1980 and int(token)<=datetime.now().year)




    ################################### Dictionary Features ################################
    def f_is_in_namelist(self, idx):
        token = self.tokens[idx].encode('ascii', 'ignore')
        if len(token) == 0:
            return 0
        return int(token.lower().strip() in self.NAME_LIST)

    def f_is_ordinal(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(token.lower().strip() in self.ORDINAL_LIST)


    # Also handled some of the common venue tokens that are also common in English???? 
    # TODO: more delicate 
    def f_is_in_venuelist(self, idx):
        token = self.tokens[idx].encode('ascii', 'ignore')
        if len(token) == 0:
            return 0
        if (idx-1) >= 0:
            prev_token = self.tokens[idx-1]
        else:
            prev_token = ''

        # Special case handling
        if token.strip() in ['In', 'Appear', 'Appears', 'Appeared', ] and len(prev_token)>0 and prev_token in ['.', ',', ';', '(', ]:
            return 1

        return int(token.lower().strip() in (self.VENUE_LIST + self.ORDINAL_LIST + self.COUNTRY_LIST) )


    ################################### Global Features ####################################

    def f_has_lbracket_before(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        if (idx-1) >= 0:
            prev_token = self.tokens[idx-1]
        else:
            return 0
        return int( prev_token in self.LBRACKET )

    def f_has_rbracket_after(self, idx):
        token = self.tokens[idx]
        if (idx+1) < self.num_tokens:
            next_token = self.tokens[idx+1]
        else:
            return 0
        return int( next_token in self.RBRACKET )

    def f_has_quotation_before(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        if (idx-1) >= 0:
            prev_token = self.tokens[idx-1]
        else:
            return 0
        return int( prev_token in self.QUOTATIONS )

    def f_has_quotation_after(self, idx):
        token = self.tokens[idx]
        if (idx+1) < self.num_tokens:
            next_token = self.tokens[idx+1]
        else:
            return 0
        return int( next_token in self.QUOTATIONS )


    #????
    def f_is_possible_volume(self, idx):
        token = self.tokens[idx]
        if ((idx-1) >=0) and ((idx+1)<self.num_tokens):
            prev_token = self.tokens[idx-1]
            next_token = self.tokens[idx+1]
            return int(prev_token in self.LBRACKET and next_token in self.RBRACKET and token.isdigit())
        else:
            return 0

    # ???? necessary?
    def f_is_at_second_half_of_string(self, idx):
        token = self.tokens[idx]
        return int(idx > self.num_tokens/2)

    def f_has_delimiter_before(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        if (idx-1) >= 0:
            prev_token = self.tokens[idx-1]
        else:
            return 0
        return int(len(prev_token)==1 and prev_token in self.DELIMITERS)

    def f_has_delimiter_after(self, idx):
        token = self.tokens[idx]
        if (idx+1) < self.num_tokens:
            next_token = self.tokens[idx+1]
        else:
            return 0
        return int( len(next_token)==1 and next_token in self.DELIMITERS)

    #????
    def f_is_an_and_between_two_names(self, idx):
        token = self.tokens[idx]
        if (idx+1) < self.num_tokens and (idx-1)>=0:
            next_token = self.tokens[idx+1]
            prev_token = self.tokens[idx-1]
        else:
            return 0
        return int(token.strip().lower()=='and' and self.f_is_capitalized(idx-1) and (self.f_is_english(idx-1)==0))

    def f_is_followed_by_year(self, idx):
        token = self.tokens[idx]
        if (idx+1) < self.num_tokens:
            next_token = self.tokens[idx+1]
        else:
            return 0
        return int((len(next_token)==2 or len(next_token)==4) and next_token.isdigit() and not token.isdigit())

    # Addressing the possible new notions in the title of publications
    def f_is_possible_new_notion(self, idx):
        token = self.tokens[idx]
        if (idx+2) < self.num_tokens:
            next_token = self.tokens[idx+1]
            next_next_token = self.tokens[idx+2]
        else:
            return 0
        p1 = re.compile(r'^[A-Z][a-z0-9]+[A-Z][a-z0-9]+$', re.MULTILINE)
        p2 = re.compile(r'^[A-Z][a-z0-9]+$', re.MULTILINE)
        p3 = re.compile(r'^[A-Z][a-z0-9]+[A-Z][a-z0-9]+[A-Z][a-z0-9]+$', re.MULTILINE)
        p4 = re.compile(r'^[a-z0-9]+$', re.MULTILINE)
        p5 = re.compile(r'[A-Z]*[A-Za-z]+-[A-Za-z]+')    #specific terminology ???? content-aware; Group-By
        # Xxxxxx, XxxxxXxxxx, XxxxXxxxXxxx, Xxxx xxxx, Xxxx Xxxx, XXXX

        pattern_1 = token.isupper() and next_token==':'
        pattern_2 = (p1.match(token) is not None) and next_token==':'
        pattern_3 = (p2.match(token) is not None) and next_token==':'
        pattern_4 = (p3.match(token) is not None) and next_token==':'
        pattern_5 = (p2.match(token) is not None) and (p2.match(next_token) is not None) and next_next_token==':'
        pattern_6 = (p2.match(token) is not None) and (p4.match(next_token) is not None) and next_next_token==':'
        pattern_7 = p5.match(token) is not None

        return int(pattern_1 or pattern_2 or pattern_3 or pattern_4 or pattern_5 or pattern_6 or pattern_7)


    def f_is_possible_boundary(self, idx):  #check if period.  Pending feature
        token = self.tokens[idx]
        if (idx+1) < self.num_tokens and (idx-1)>=0:
            next_token = self.tokens[idx+1]
            prev_token = self.tokens[idx-1]
        else:
            return 0

        return int( (token == '.' and prev_token.islower() and next_token[0].isupper()) or 
                    (token[-1]=='.' and token[0].islower() and next_token[0].isupper()) 
                  )


    # Ranging features: May need to come up with correct threshold after experiments
    # def f_is_len_in_






if __name__ == '__main__':
    pass