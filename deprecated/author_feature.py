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
class AuthorFeatureBuilder(FeatureBuilder):
    def __init__(self, record):
        super(AuthorFeatureBuilder, self).__init__()
        self.record = record
        self.tokens = Tokens(record).tokens
        self.num_tokens = len(self.tokens)
        self.features = None      # list of list of features for every name; e.g. [[1,1,1,1],[...], ...]
        
        self.NUM_REGEX = re.compile('\d')
        self.DELIMITERS = [',', '.', ';', ]
        self.NAME_LIST = [item.strip() for item in open('data/name.lst','r').readlines()]

        self.pipeline = [
            'f_is_capitalized',
            'f_is_all_upper',
            'f_is_english',
            'f_is_punctuation',
            'f_is_sequential_punctuation',
            'f_has_digit',
            'f_is_all_digit',
            'f_is_in_namelist',
            'f_is_fname_abbrev',
            'f_is_preceeded_by_delimiter',
            'f_is_followed_by_delimiter',
            'f_is_an_and_between_two_names',
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

        #????
        # num_tokens = float(self.tokens_obj.length)
        # for token in tokens:
        #     for j in range(self.num_features)
        #         action = self.pipeline[j]
        #         feature_accumulator[j] += self.action(token)
        # for tmp_feature in feature_accumulator:
        #     features.append(tmp_feature / num_tokens)
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

    def f_is_punctuation(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(len(token) == 1 and token in punctuation)

    def f_is_sequential_punctuation(self, idx): #e.g. K.C.-C. Chang
        token = self.tokens[idx]
        if len(token) <= 1:
            return 0
        ret = 1
        for t in token:
            if t not in punctuation:
                ret = 0
                break
        return ret

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

    def f_is_in_namelist(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        return int(token.lower().strip() in self.NAME_LIST)


    ################################### Global Features ####################################
    # Is single capitalized alphabet, and followed by a period
    def f_is_fname_abbrev(self, idx):
        token = self.tokens[idx]
        if len(token) == 0:
            return 0
        # if (idx+1) < self.num_tokens:
        #     next_token = self.tokens[idx+1]
        # else:
        #     return 0
        return int(len(token) == 1 and token.isupper())
        # return int( (len(token) == 1 and token.isupper() and next_token=='.') )

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

    def f_is_an_and_between_two_names(self, idx):
        token = self.tokens[idx]
        if (idx+1) < self.num_tokens and (idx-1)>=0:
            next_token = self.tokens[idx+1]
            prev_token = self.tokens[idx-1]
        else:
            return 0

        return int(token.strip().lower()=='and' and self.f_is_capitalized(idx-1) and (self.f_is_english(idx-1)==0))
            


    # POS tagging pass?




if __name__ == '__main__':
    seg = 'RoundTripRank: Graph-based Proximity with Importance and Specificity. Y. Fang, K. C.-C. Chang, and H. W. Lauw. In ICDE, 2013.'
    f = AuthorFeatureBuilder(seg)
    print f.print_features()