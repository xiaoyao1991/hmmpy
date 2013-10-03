import numpy as np
from chmm import *
from utils import get_binary_vector, log_err
from training_set_generator import get_training_samples_BW

FEATURE_DIM = 20    # ???? get rid of one of the feature which is 
FEATURE_SPACE = 2**FEATURE_DIM

MAX_ITER = 10
MIN_IMPROVMENT = 0.1

# 1. Read in the data
log_err('Reading data from retrieval...')
data = get_training_samples_BW('http://scholar.google.com/citations?user=YU-baPIAAAAJ&hl=en', True)

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


# 4. Use chmm to construct initialize model before training.
log_err('Building draft models')
V = [i for i in range(FEATURE_SPACE)]
Pi = np.array([0.5,0.5,])
A = np.array([[0.5, 0.5,],[0.5, 0.5,],])
B = np.array([[1.0/FEATURE_SPACE]*FEATURE_SPACE, [1.0/FEATURE_SPACE]*FEATURE_SPACE, ])

HMMauthor = HMM(2, A=A, B=B, V=V)
HMMtitle = HMM(2, A=A, B=B, V=V)
HMMvenue = HMM(2, A=A, B=B, V=V)

# 5. Train
log_err('Training author model...')
baum_welch(HMMauthor, encoded_author_sequences,verbose=True)
log_err('Training title model...')
baum_welch(HMMtitle, encoded_title_sequences,verbose=True)
log_err('Training venue model...')
baum_welch(HMMvenue, encoded_venue_sequences,verbose=True)

print '\nauthor model'
print HMMauthor

print '\ntitle model'
print HMMtitle

print '\nvenue model'
print HMMvenue