from hmm import HMM
from feature import FeatureGenerator
from utils import get_binary_vector, log_err
from tokens import Tokens
from training_set_generator import get_training_samples_venue, get_training_samples_author, get_training_samples
import sys
import pickle
import operator

from retrainer import Retrainer


class HMMClassifier(object):
    def __init__(self, **kwarg):    # lname, url, other prior knowledge
        super(HMMClassifier, self).__init__()
        self.HMMauthor = HMM('author', 2)
        self.HMMvenue = HMM('venue', 2)     # Not important
        self.HMMentire = HMM('entire', 6)   # Set empirically
        self.observations_raw = []
        self.observation_sequences = []
        self.labels = []


    def predict(self, segment):
        author_likelihood = self.HMMauthor.evaluate(segment)
        venue_likelihood = self.HMMvenue.evaluate(segment)
        print segment
        print 'author likelihood:\t' , author_likelihood
        print 'venue likelihood:\t' , venue_likelihood

    def decode(self, segment):
        # print segment
        observation_sequence, decoded_sequence = self.HMMentire.decode(segment)
        
        self.observations_raw.append(segment)
        self.observation_sequences.append(observation_sequence)
        self.labels.append(decoded_sequence)


        # segment the labeling into parts
        author_field = []
        title_field = []
        venue_field = []
        year_field = []
        raw_tokens = Tokens(segment).tokens
        for i in range(len(decoded_sequence)):
            token_i = raw_tokens[i]
            label_i = decoded_sequence[i]
            if label_i in [0,1]:
                author_field.append(token_i)
            if label_i == 2:
                continue
            if label_i == 3:
                title_field.append(token_i)
            if label_i == 4:
                venue_field.append(token_i)
            if label_i == 5:
                year_field.append(token_i)

        return ' '.join(author_field), ' '.join(title_field), ' '.join(venue_field), list(set(year_field))
        # Additional step: to calculate the overall sum of P(X1|FN,LN,DL...) + P(X2|TI,TI,TI...) + P(X3|VN,VN,VN...) + P(X4|DT)
        # 1. Find boundaries: 
        # boundaries = [[], [], []]
        # label_ranges = [[0,1,2], [3], [2,4,5]]
        # for i in range(len(label_ranges)):
        #     label_range = label_ranges[i]
        #     for j in range(len(decoded_sequence)):
        #         if decoded_sequence[j] in label_range:
        #             boundaries[i].append()

    
    def decode_without_constraints(self, segment):
        print segment
        observation_sequence, decoded_sequence = self.HMMentire.decode_without_constraints(segment)
        
        self.observations_raw.append(segment)
        self.observation_sequences.append(observation_sequence)
        self.labels.append(decoded_sequence)

        for vector, decoding, token in zip(observation_sequence, decoded_sequence, Tokens(segment).tokens):
            if decoding == 0:
                label = 'FN'
            elif decoding == 1:
                label = 'LN'
            elif decoding == 2:
                label = 'DL'
            elif decoding == 3:
                label = 'TI'
            elif decoding == 4:
                label = 'VN'
            elif decoding == 5:
                label = 'YR'
            else:
                label = str(decoding) + ', PROBLEM'
            print vector, '\t', label, '\t', token
        print '\n\n'

    def cross_correct(self):
        absolute_correct = []
        absolute_wrong = []

        # 1. Confirm what's the big structure of the publication inside this specific domain
        counter = {}
        for l in self.labels:
            first_label = str(l[0])
            if counter.has_key(first_label):
                counter[first_label] += 1
            else:
                counter[first_label] = 1
        sorted_counter = sorted(counter.iteritems(), key=operator.itemgetter(1), reverse=True)
        print 'First labels distribution: ', sorted_counter


    def serialize(self):
        fp = open('hmmc.pkl', 'wb')
        pickle.dump(self, fp, -1)
        fp.close()
