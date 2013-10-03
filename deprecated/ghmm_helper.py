import pickle
import ghmm
from utils import get_binary_vector, log_err
from ghmm import *

FEATURE_DIM = 20    # ???? get rid of one of the feature which is 
FEATURE_SPACE = 2**FEATURE_DIM

MAX_ITER = 10
MIN_IMPROVMENT = 0.1

# 1. Read in the data from Mac
log_err('Reading data from pickle...')
fp = open('data.pkl', 'rb')
data = pickle.load(fp)
fp.close()

# 2. Create binary-decimal mapper
log_err('Start generating binary vectors...')
feature_symbol_mapping = {}
possible_features = get_binary_vector(FEATURE_SPACE, FEATURE_DIM)
log_err('Start mapping feature vectors to int encoding...')
for possible_feature in possible_features:
    feature_symbol_mapping[str(possible_feature)] = len(feature_symbol_mapping)
log_err('Finish mapping...')

# 3. Map the features from bianry to decimal
log_err('Converting the features')
encoded_author_sequences = []
encoded_title_sequences = []
encoded_venue_sequences = []

for sample in data['author_sequences']:
    tmp = []
    for vector in sample:
        tmp.append(feature_symbol_mapping[str(vector)])
    encoded_author_sequences.append(tmp)

for sample in data['title_sequences']:
    tmp = []
    for vector in sample:
        tmp.append(feature_symbol_mapping[str(vector)])
    encoded_title_sequences.append(tmp)

for sample in data['venue_sequences']:
    tmp = []
    for vector in sample:
        tmp.append(feature_symbol_mapping[str(vector)])
    encoded_venue_sequences.append(tmp)

# 4. Convert the raw format of data to ghmm recognizable training set
log_err('Generating SequenceSet...')
sigma = IntegerRange(0, FEATURE_SPACE)

author_seq_set = SequenceSet(sigma, encoded_author_sequences)
title_seq_set = SequenceSet(sigma, encoded_title_sequences)
venue_seq_set = SequenceSet(sigma, encoded_venue_sequences)

# 5. Use GHMM to construct initialize model before training.
log_err('Building draft models')
pi = [0.5,0.5,]
A = [
        [0.5, 0.5,],
        [0.5, 0.5,],
    ]
B = [
        [1.0/FEATURE_SPACE]*FEATURE_SPACE, 
        [1.0/FEATURE_SPACE]*FEATURE_SPACE, 
    ]

HMMauthor = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)
HMMtitle = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)
HMMvenue = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)

# 6. Train      ???? Segfault when training
log_err('Training...')
HMMauthor.baumWelch(author_seq_set, MAX_ITER, MIN_IMPROVMENT)
log_err('>>>>>>>>>>>bp1')
HMMtitle.baumWelch(title_seq_set, MAX_ITER, MIN_IMPROVMENT)
log_err('>>>>>>>>>>>bp2')
HMMvenue.baumWelch(venue_seq_set, MAX_ITER, MIN_IMPROVMENT)
log_err('>>>>>>>>>>>bp3')

print '\nauthor model'
print HMMauthor

print '\ntitle model'
print HMMtitle

print '\nvenue model'
print HMMvenue