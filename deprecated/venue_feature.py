import re
import sys
from feature import FeatureBuilder
from string import punctuation, ascii_uppercase
from feature import FeatureBuilder
from tokens import Tokens
import nltk

# DEPRECATED!

# Takes a record segment, tokenize it to token list, and generate features for it.
# EVERY SEGMENT HAS ONE FEATURE VECTOR
class VenueFeatureBuilder(FeatureBuilder):
    def __init__(self, record):
        super(VenueFeatureBuilder, self).__init__()
        self.record = record
        self.tokens = Tokens(record).tokens
        self.num_tokens = len(self.tokens)
        self.features = None
        
        self.NUM_REGEX = re.compile('\d')
        self.CHAR_DIGIT_MIX_REGEX = re.compile('((^[a-zA-Z]+\d{4}$)|(^[a-zA-Z]+\d{2}$))|((^\d{4}[a-zA-Z]+$)|(^\d{2}[a-zA-Z]+$))', re.MULTILINE)
        
        self.DELIMITERS = [',', '.', ';', ]
        self.VENUE_LIST = [item.strip() for item in open('data/venue.lst','r').readlines()]
        self.ORDINAL_LIST = [item.strip() for item in open('data/ordinal.lst','r').readlines()]

        self.pipeline = [
            'f_is_capitalized',
            'f_is_all_upper',
            'f_is_english',
            'f_has_both_char_and_digit',
            'f_is_ordinal',
            'f_is_punctuation',
            'f_has_digit',
            'f_is_all_digit',
            'f_is_in_venuelist',
            'f_is_preceeded_by_delimiter',
            'f_is_followed_by_delimiter',
            'f_is_followed_by_year',
        ]

        self.build()



    def build(self):
        features = []

        for i in range(self.num_tokens):
            sub_features = []
            for pipe in self.pipeline:
                action = getattr(self, pipe)
                sub_features.append(action(i))
            features.append(sub_features)

        self.features = features


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

    def f_is_ordinal(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(token.lower().strip() in self.ORDINAL_LIST)

    def f_is_punctuation(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(len(token) == 1 and token in punctuation)

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

    def f_is_in_venuelist(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(token.lower().strip() in self.VENUE_LIST)


    ################################### Global Features ####################################
    def f_is_preceeded_by_delimiter(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        if (idx-1) >= 0:
            prev_token = self.tokens[idx-1]
        else:
            return 0
        return int(len(prev_token)==1 and prev_token in self.DELIMITERS)

    def f_is_followed_by_delimiter(self, idx):
        token = self.tokens[idx]
        if (idx+1) < self.num_tokens:
            next_token = self.tokens[idx+1]
        else:
            return 0
        return int( len(next_token)==1 and next_token in self.DELIMITERS)

    def f_is_followed_by_year(self, idx):
        token = self.tokens[idx]
        if (idx+1) < self.num_tokens:
            next_token = self.tokens[idx+1]
        else:
            return 0
        return int((len(next_token)==2 or len(next_token)==4) and next_token.isdigit())

    # POS tagging pass?




if __name__ == '__main__':
    seg = 'RoundTripRank: Graph-based Proximity with Importance and Specificity. Y. Fang, K. C.-C. Chang, and H. W. Lauw. In ICDE, 2013.'
    f = VenueFeatureBuilder(seg)
    print f.print_features()