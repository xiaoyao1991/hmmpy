import numpy as np
import hmm_faster
from math import log
from utils import get_binary_vector, log_err
from training_set_generator import get_training_samples_BW

FEATURE_DIM = 20    # ???? get rid of one of the feature which is 
FEATURE_SPACE = 2**FEATURE_DIM

MAX_ITER = 10
MIN_IMPROVMENT = 0.1

# 1. Read in the data
log_err('Reading data from retrieval...')
# data = get_training_samples_BW('http://scholar.google.com/citations?user=YU-baPIAAAAJ&hl=en', True)
data = get_training_samples_BW('http://scholar.google.com/citations?user=x3LTjz0AAAAJ&hl=en', True)

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


# 4. Use hmmpytk to construct initialize model before training.
log_err('Building draft models')
states = ['0', '1']
observation_list = [str(i) for i in range(FEATURE_SPACE)]
pi = {'0': log(0.5), '1':log(0.5)}
A = {'0': {'0': log(0.5), '1':log(0.5)}, '1':{'0':log(0.5), '1':log(0.5)}}

B = {'0':{}, '1':{}}
for i in range(FEATURE_SPACE):
    B['0'][str(i)] = log(1.0/FEATURE_SPACE)
    B['1'][str(i)] = log(1.0/FEATURE_SPACE)

HMMauthor = hmm_faster.HMM()
HMMtitle = hmm_faster.HMM()
HMMvenue = hmm_faster.HMM()

HMMauthor.set_states(states)
HMMauthor.set_observations(observation_list)
HMMauthor.set_initial_matrix(pi)
HMMauthor.set_transition_matrix(A)
HMMauthor.set_emission_matrix(B)

HMMtitle.set_states(states)
HMMtitle.set_observations(observation_list)
HMMtitle.set_initial_matrix(pi)
HMMtitle.set_transition_matrix(A)
HMMtitle.set_emission_matrix(B)

HMMvenue.set_states(states)
HMMvenue.set_observations(observation_list)
HMMvenue.set_initial_matrix(pi)
HMMvenue.set_transition_matrix(A)
HMMvenue.set_emission_matrix(B)

# 5. Train
log_err('Training author model...')
print 'author model'
for training_seq in encoded_author_sequences:
    log_err('\tauthor training...')
    HMMauthor.train([str(t) for t in training_seq], max_iteration=1)
    print HMMauthor.get_initial_matrix()
    print HMMauthor.get_transition_matrix()
print '\n\n'

log_err('Training title model...')
print 'title model'
for training_seq in encoded_title_sequences:
    log_err('\ttitle training...')
    HMMtitle.train([str(t) for t in training_seq], max_iteration=1)
    print HMMtitle.get_initial_matrix()
    print HMMtitle.get_transition_matrix()
print '\n\n'

log_err('Training venue model...')
print 'venue model'
for training_seq in encoded_venue_sequences:
    log_err('\tvenue training...')
    HMMvenue.train([str(t) for t in training_seq], max_iteration=1)
    print HMMvenue.get_initial_matrix()
    print HMMvenue.get_transition_matrix()