from feature import *

PARALLEL_PIPELINE = [
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

                        # Optional features, parallel
                        'f_is_repeated_name_token',
                        'f_is_repeated_delimiter_token',
                        'f_is_repeated_title_token',
                        'f_is_repeated_venue_token',
]


# This is the feature set used during second iteration, inherited from FeatureGenerator
# @param: parallel params means the segmentation results from the first iteration
class BoostingFeatureGenerator(FeatureGenerator):
    def __init__(self, token_BGM, pattern_BGM):
        super(BoostingFeatureGenerator, self).__init__()
        
        self.token_BGM = token_BGM
        self.pattern_BGM = pattern_BGM
        self.pipeline = PARALLEL_PIPELINE


    # ================================= Token level, parallel features. =================================
    # Check the most labeled label for this token, and check the total occurance
    def f_is_repeated_name_token(self, idx):
        token = self.tokens[idx]
        if self.token_BGM.has_key(token):
            total_occurance = 0
            for k,v in self.token_BGM[token]:
                total_occurance += v
            return int(total_occurance>=5 and self.token_BGM[token][0][0] in ['0', '1'])
        else:
            return 0


    def f_is_repeated_delimiter_token(self, idx):
        token = self.tokens[idx]
        if self.token_BGM.has_key(token):
            total_occurance = 0
            for k,v in self.token_BGM[token]:
                total_occurance += v
            return  int(total_occurance>=5 and self.token_BGM[token][0][0] == '2')
        else:
            return 0


    def f_is_repeated_title_token(self, idx):
        token = self.tokens[idx]
        if self.token_BGM.has_key(token):
            total_occurance = 0
            for k,v in self.token_BGM[token]:
                total_occurance += v
            return int(total_occurance>=2 and self.token_BGM[token][0][0] == '3')
        else:
            return 0


    def f_is_repeated_venue_token(self, idx):
        token = self.tokens[idx]
        if self.token_BGM.has_key(token):
            total_occurance = 0
            for k,v in self.token_BGM[token]:
                total_occurance += v
            return int(total_occurance>=5 and self.token_BGM[token][0][0] == '4')
        else:
            return 0

