import math, sys
from utils import log_err, log_1p, log_sum, deprecated, vec_to_int, INT_LABEL_MAP, INT_LABEL_MAP_SIX
from string import punctuation
from datetime import datetime
from tokens import Tokens
from feature import FeatureGenerator
from boosting_feature import BoostingFeatureGenerator
from training_set_generator import get_training_samples, get_training_samples_raw_cora
from language_model import FeatureEntity, FeatureEntityList, LanguageModel
import numpy as np
import operator
from scipy.sparse import *
from scipy import *
import pickle
from sparse_matrix_dict import SparseMatrixDict
import time

class HMM(object):
    """HMM"""
    def __init__(self, name, states, emissions=0, useLaplaceRule=True):
        super(HMM, self).__init__()
        
        self.name = name
        self.pi = None
        self.transitions = None
        self.emissions = None
        self.log_pi = None
        self.log_transitions = None
        self.log_emissions = None
        self.useLaplaceRule = useLaplaceRule

        self.observation_sequences = None   # Keep the training samples, so that when new correct record comes in, can retrain the model to make it more robust
        self.label_sequences = None         

        self.training_sample_size = 0
        self.feature_dim = 0    # Feature dimension

        self.state_dim = states
        self.observation_dim = 2    #binary features, dim=2
        self.encoded_observation_dim = emissions    # decimal symbol/observation dimension
        self.feature_symbol_mapping = {}

        self.feature_entity_list = FeatureEntityList()  # Form the background model
        self.feature_generator = None # Form the feature generator

        self.context_model = []

        self.punctuation = ['!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~',]
        self.possible_DL = [',', ';', ').', '.)', '".', '."', ')."', ')".', '.)"']



    def set_pi(self, pi):
        if sum(pi) > 0:
            self.pi = pi
            self.log_pi = []
            for i in xrange(len(pi)):
                self.log_pi.append(math.log(pi[i]))
        else:
            self.log_pi = pi
            self.pi = []
            for i in xrange(len(pi)):
                self.pi.append(math.exp(pi[i]))

    # TO-DO: Need to check if overflow/underflow
    def set_transitions(self, transitions):
        shape = (len(transitions), len(transitions))
        if sum(transitions[0]) > 0:
            self.transitions = transitions
            self.log_transitions = np.zeros(shape=shape)
            for i in xrange(len(transitions)):
                for j in xrange(len(transitions)):
                    self.log_transitions[i][j] = math.log(transitions[i][j])
        else:
            self.log_transitions = transitions
            self.transitions = np.zeros(shape=shape)
            for i in xrange(len(transitions)):
                for j in xrange(len(transitions)):
                    self.transitions[i][j] = math.exp(transitions[i][j])

    def set_emissions(self, emissions):
        shape = (len(emissions), len(emissions[0]))
        if sum(emissions[0]) > 0:
            self.emissions = emissions
            self.log_emissions = np.zeros(shape=shape)
            for i in xrange(len(emissions)):
                for j in xrange(len(emissions[0])):
                    self.log_emissions[i,j] = math.log(emissions[i][j])
        else:
            self.log_emissions = emissions
            self.emissions = np.zeros(shape=shape)
            for i in xrange(len(emissions)):
                for j in xrange(len(emissions[0])):
                    self.emissions[i][j] = math.exp(emissions[i][j])

    def get_log_emission(self, i,j, useLaplaceRule):
        if useLaplaceRule:
            row_sum_i = 0.0     #denominator
            row_i = self.emissions_counter.getrow(i)
            for k in row_i.nonzero()[1]:
                row_sum_i += row_i[0,k]
            row_sum_i += 1.0 * self.encoded_observation_dim

            return math.log(row_i[0,j] + 1.0) - math.log(row_sum_i)
        

        else:
            
            row_i = self.emissions_counter.getrow(i)
            if row_i[0,j] == 0.0:
                return -10000   #???? proper
            else:
                row_sum_i = 0.0     #denominator
                for k in row_i.nonzero()[1]:
                    row_sum_i += row_i[0,k]

                return math.log(row_i[0,j]) - math.log(row_sum_i)

    # Supervised Learning using the maximum likelihood method
    # observation_sequences: 
    # [
    #     [[1,0,0,1], [0,1,1,1], ....],
    #     [...],
    #     ...
    # ]
    def train(self, observation_sequences, label_sequences, useLaplaceRule=True):
        # Setup information
        self.observation_sequences = observation_sequences
        self.label_sequences = label_sequences
        self.training_sample_size = len(label_sequences)
        self.feature_dim = len(observation_sequences[0][0])  # feature dimension
        self.encoded_observation_dim = self.observation_dim ** self.feature_dim

        log_err('\tBuilding Background Knowledge Model...')
        for observation_sequence, label_sequence in zip(observation_sequences, label_sequences):
            for feature_vector, label in zip(observation_sequence, label_sequence):
                self.feature_entity_list.add_entity(feature_vector, label)

        #????delete feature map

        # Initial setup
        pi = np.zeros(self.state_dim)
        transitions = np.zeros((self.state_dim, self.state_dim))
        emissions = lil_matrix((self.state_dim, self.encoded_observation_dim))

        pi_counter = np.zeros(self.state_dim)
        transitions_counter = np.zeros((self.state_dim, self.state_dim))
        # self.emissions_counter = lil_matrix((self.state_dim, self.encoded_observation_dim))
        emissions_counter = SparseMatrixDict((self.state_dim, self.encoded_observation_dim), log_result=False, laplace_smoothing=useLaplaceRule)

        log_pi = np.zeros(self.state_dim)
        log_transitions = np.zeros((self.state_dim, self.state_dim))

        log_err('\tStart counting pi...')
        # 1. Counting first state in every label sequence to form pi
        for label_sequence in self.label_sequences:
            pi_counter[label_sequence[0]] += 1.0

        log_err('\tStart counting transitions...')
        # 2. Count all state transitions
        for label_sequence in self.label_sequences:
            for j in xrange(1, len(label_sequence)):
                transitions_counter[label_sequence[j-1]][label_sequence[j]] += 1.0

        log_err('\tStart counting emissions...')
        # 3. Count emissions for each label
        for i in xrange(self.training_sample_size):
            for j in xrange(len(self.observation_sequences[i])):
                symbol = vec_to_int(self.observation_sequences[i][j])    # encode the feature vector into int
                emissions_counter[self.label_sequences[i][j], symbol] += 1.0

        log_err('\tStart forming log probability...')

        # 4. Form log probability, using Laplace correction to avoid zero probabilities by default
        if useLaplaceRule:
            for i in xrange(self.state_dim):
                pi_counter[i] += 1.0

                for j in xrange(self.state_dim):
                    transitions_counter[i][j] += 1.0
                log_err('\t\t[!]Laplace normalize emission...')
                # for k in xrange(self.encoded_observation_dim):  #????need handle
                #     emissions_counter[i,k] += 1.0
                # log_err('\t\t[!]Laplace normalize complete!!!!')

        pi_count = sum(pi_counter)
        transition_count = [sum(transition) for transition in transitions_counter]
        
        # log_err('\t\tCalculating emission counts')
        # emission_count = []
        # for i in xrange(self.state_dim):
        #     tmp_sum = 0.0
        #     row_i = emissions_counter.getrow(i)
        #     for j in row_i.nonzero()[1]:
        #         tmp_sum += row_i[0,j]
            
        #     if tmp_sum > 0.0:
        #         emission_count.append(tmp_sum)
        #     else:
        #         emission_count.append(1.0)  #????avoid division by zero
        
        log_err('\tStart computing probability...')
        for i in xrange(len(pi_counter)):
            pi[i] = pi_counter[i] / pi_count
            if pi_counter[i] == 0.0:
                log_pi[i] = -10000
            else:
                log_pi[i] = math.log(pi_counter[i] / pi_count)
        log_err('\t\tPi complete!')
        for i in xrange(self.state_dim):
            for j in xrange(self.state_dim):
                transitions[i][j] = transitions_counter[i][j] / transition_count[i]
                if transitions_counter[i][j] == 0.0:
                    log_transitions[i][j] = -10000
                else:
                    log_transitions[i][j] = math.log(transitions_counter[i][j] / transition_count[i])
        log_err('\t\tTransitions complete!')
        # for i in xrange(self.state_dim):
        #     for j in xrange(self.encoded_observation_dim):
        #         emissions[i,j] = emissions_counter[i,j] / emission_count[i]
        #         if emissions_counter[i,j] == 0.0:
        #             log_emissions[i,j] = -10000     #????
        #         else:
        #             log_emissions[i,j] = math.log(emissions_counter[i,j] / emission_count[i])
        log_err('\t\tEmissions complete!')

        self.pi = pi
        self.transitions = transitions
        self.emissions = emissions
        self.log_pi = log_pi
        self.log_transitions = log_transitions

        emissions_counter.change_state()
        self.log_emissions = emissions_counter

        self.log_emissions.print_stats()

    def print_model(self):
        print 'Pi:\n' , self.pi
        print 'Transition Prb:\n', self.transitions
        print 'Emission Prb:\n', self.emissions
        print 'Log Pi:\n' , self.log_pi            
        print 'Log Transition Prb:\n', self.log_transitions


    # Ugly???????????????????????
    def decode(self, raw_segment, partial_features=False, use_boost_feature=False, token_BGM=None, pattern_BGM=None):
        # 1. Translate token list to feature vectors
        if not use_boost_feature:
            self.feature_generator = FeatureGenerator()
        else:
            self.feature_generator = BoostingFeatureGenerator(token_BGM, pattern_BGM)
        
        observation_sequence = self.feature_generator.build(raw_segment)  #20131002
        observation_tokens = self.feature_generator.tokens

        # 2. Encode the feature vectors into integers
        encoded_sequence = []
        for vector in observation_sequence:
            encoded_sequence.append(vec_to_int(vector))

        # 3. Call the viterbi algorithm
        decoded_label_sequence = self.viterbi_with_constraint(encoded_sequence, observation_tokens)
        for (observation_vector, token, label) in zip(observation_sequence, observation_tokens, decoded_label_sequence):
            print observation_vector, '\t', token, '\t', INT_LABEL_MAP_SIX[label]
        print '====================================================================\n\n'

        return observation_sequence, decoded_label_sequence

    def decode_without_constraints(self, raw_segment):
        log_err('[...] Decoding ' + raw_segment)

        # 1. Translate token list to feature vectors
        self.feature_generator = FeatureGenerator()

        observation_sequence = self.feature_generator.build(raw_segment)
        observation_tokens = self.feature_generator.tokens

        # 2. Encode the feature vectors into integers
        encoded_sequence = []
        for vector in observation_sequence:
            encoded_sequence.append(vec_to_int(vector))

        # 3. Call the viterbi algorithm
        decoded_label_sequence = self.viterbi(encoded_sequence)
        for (observation_vector, token, label) in zip(observation_sequence, observation_tokens, decoded_label_sequence):
            print observation_vector, '\t', token, '\t', INT_LABEL_MAP_SIX[label]
        print '====================================================================\n\n'

        return observation_sequence, decoded_label_sequence


    # Given the observation sequence, find the most likely hidden sequence
    def viterbi(self, encoded_sequence):
        log_V = [{}]
        path = {}

        # 1. Initialize at t = 0
        for i in xrange(self.state_dim):
            # log_V[0][str(i)] = self.log_pi[i] + self.get_log_emission(i,encoded_sequence[0], self.useLaplaceRule)
            log_V[0][str(i)] = self.log_pi[i] + self.log_emissions[i,encoded_sequence[0]]
            path[str(i)] = [i]

        # 2. Run viterbi for t > 0
        for t in xrange(1, len(encoded_sequence)):
            log_V.append({})
            new_path = {}

            for i in xrange(self.state_dim):
                (prob, state) = max([(log_V[t-1][str(j)] + self.log_transitions[j][i] + self.log_emissions[i,encoded_sequence[t]], j) for j in xrange(self.state_dim)])
                log_V[t][str(i)] = prob
                new_path[str(i)] = path[str(state)] + [i]
            path = new_path

        (prob, state) = max([(log_V[len(encoded_sequence)-1][str(j)], j) for j in xrange(self.state_dim)])
        return path[str(state)]


    # Return the best path under some constraints
    def viterbi_with_constraint(self, encoded_sequence, tokens):
        log_V = [{}]
        path = {}
        DL_separated_names_flag = True  #Default to be True

        # 1. Initialize at t = 0
        for i in xrange(self.state_dim):
            log_V[0][str(i)] = self.log_pi[i] + self.log_emissions[i,encoded_sequence[0]]
            path[str(i)] = [i]

        # 2. Run viterbi for t > 0
        for t in xrange(1, len(encoded_sequence)):
            log_V.append({})
            new_path = {}

            # print tokens[:t]
            # print sorted(log_V[-2].iteritems(), key=operator.itemgetter(1), reverse=True)
            # print path
            # print '\n'
            for i in xrange(self.state_dim):
                log_counter = {}
                for j in xrange(self.state_dim):
                    log_counter[str(j)] = log_V[t-1][str(j)] + self.log_transitions[j][i] + self.log_emissions[i,encoded_sequence[t]]



                # max[Pa * P(sunny|sunny), Pb * P(sunny|cloudy)...] --> just for sunny, 
                # AND SEE WHICH PRODUCT IS THE LARGEST, THEN DECIDE THAT STATE AS THE PREVIOUS STATE, THUS RECURSIVELY DECIDE ON THE WHOLE SEQUENCE
                max_list = sorted(log_counter.iteritems(), key=operator.itemgetter(1), reverse=True)
                # print max_list

                # Looping through the max's, from the largest to smallest
                valid_flag = False
                name_penalty_flag = False
                for k in xrange(len(max_list)):
                    (state, prob) = max_list[k]
                    prev_states = path[str(state)]

                    # Apply constraint:
                    # if the new state is in previous states:
                    # 1. if the last state is delimiter, then check if the new state is the second last state, if not, then not allow
                    # 2. if the last state is not delimiter, then check if the new state is the last state, if not, then not allow
                    # 3. 0,1 states(FN, LN) states need special consideration
                    if i not in [0,1]:
                        last_occurence = ''.join([str(c) for c in prev_states]).rfind(str(i))
                    elif i in [0,1]:
                        last_occurence_0 = ''.join([str(c) for c in prev_states]).rfind('0')
                        last_occurence_1 = ''.join([str(c) for c in prev_states]).rfind('1')
                        last_occurence = max(last_occurence_0, last_occurence_1)

                    if last_occurence == -1 or last_occurence == (len(prev_states)-1):
                        if i == 2 and tokens[t] not in self.punctuation+self.possible_DL :   # Hardcode rule [2013/07/18]: DL cannot be word
                            continue
                        if i == 4 and prev_states[-1] == 3: #Hardcode rule [2013/07/23]: For VN consideration
                            continue
                        if i == 5 and (tokens[t].isdigit() is False):
                            continue
                        valid_flag = True
                        break

                    # Name position constraints
                    elif i in [0,1]:    # Loosen this constraint, cuz now we see more name patterns
                        # ???? Name Constraint Proposal: As long as the first state in the prev state that is not 2, is in [0,1], then any 0,1 is allowed
                        # if prev_states[-1] == 2 and prev_states[-2] in [0,1]:
                        #     valid_flag = True
                        #     break
                        # elif prev_states[-1] in [0,1]:
                        #     valid_flag = True
                        #     break
                        # else:
                        #     continue

                        # Hardcoded rule [2013/07/18]: If consecutive(8) FN & LN without DL, then probably should mark FN or LN after a DL
                        # ???? probably want to loosen this rule, when you see two names separated by DL already, even though the later name 
                        # appear to be super long, still treat it as DL serparated 
                        
                        name_start = False
                        longest_consecutive = 0
                        # print 'debug on name constraint'
                        # print 'i:',i,'\tstate:',state
                        # print 'prev_states:',prev_states
                        for prev_state in reversed(prev_states):
                            if (not name_start) and (prev_state not in [0,1]):
                                continue
                            if prev_state in [0,1]:
                                name_start = True
                                longest_consecutive += 1
                            if (name_start is True) and (prev_state not in [0,1]):
                                break
                        # print longest_consecutive ????
                        if longest_consecutive > 8:
                            DL_separated_names_flag &= False
                        else:
                            DL_separated_names_flag &= True

                        # 
                        if not DL_separated_names_flag:
                            if prev_states[-1] not in [0,1]:
                                name_penalty_flag = True
                                continue    #???? 
                            else:
                                valid_flag = True
                                break
                        #
                        sub_valid_flag = False
                        for prev_state in reversed(prev_states):
                            if prev_state != 2: # The first non-2 state in the previous states
                                if prev_state in [0,1]:
                                    sub_valid_flag = True
                                    break
                                else:
                                    sub_valid_flag = False
                                    break
                            else:
                                continue

                        #
                        if sub_valid_flag:
                            valid_flag = True
                            break
                        else:
                            continue

                    elif i == 2:
                        if tokens[t] in self.punctuation+self.possible_DL:
                            valid_flag = True
                            break
                        else:
                            continue

                    elif i == 3:
                        continue
                    elif i == 5:
                        if not tokens[t].isdigit() or int(tokens[t])<1980 or int(tokens[t])>datetime.now().year:
                            continue
                        else:
                            valid_flag = True
                            break
                    elif i == 4:
                        # Additional constraint when dealing with retrain [2013/7/23]
                        if prev_states[-1] == 3:
                            continue
                        else:
                            valid_flag = True
                            break
                        # prev_states[-1] = prev_states[-1]
                        # if prev_states[-1] == 2 and last_occurence == (len(prev_states)-2):
                        #     valid_flag = True
                        #     break
                        # else:
                        #     continue

                # print 'Max List: ', max_list
                # Update the thang
                if valid_flag:
                    log_V[t][str(i)] = prob
                    new_path[str(i)] = path[str(state)] + [i]
                    
                    # Additional constraints: 
                    # Restrict delimiter to be always delimiter; Restrict 4-digit-make-sense-number to be always date [2013/07/08]
                    if i in [0,1] and name_penalty_flag:    # penalty on name [2013/07/18]
                        log_V[t][str(i)] = prob * 1.5   #???? Empirical penalty on assignment of name label
                    if tokens[t] in self.possible_DL:
                        new_path[str(i)] = path[str(state)] + [2]
                    try:    # [Year Constraints] ???? may need more considerations, narrow the constraint
                        tmp = int(tokens[t])
                        if tmp>= 1980 and tmp<=datetime.now().year:
                            new_path[str(i)] = path[str(state)] + [5]
                    except:
                        pass
                
                else:   # When no label at this time is appropriate  ??????????
                    # print 'NOTALLOW 3 --> ', i
                    # Empiric setting: Do not penalize DL? hardcoded rule [2013/07/18]
                    if i != 2:
                        log_V[t][str(i)] = 2 * max_list[-1][1]
                        new_path[str(i)] = path[str(state)] + [i]   #???? Should append itself? 
                    else:
                        log_V[t][str(i)] = max_list[-1][1]
                        new_path[str(i)] = path[str(state)] + [i]

                    # Additional constraints: 
                    # Restrict delimiter to be always delimiter; Restrict 4-digit-make-sense-number to be always date [2013/07/08]
                    if tokens[t] in [',', ';']:
                        new_path[str(i)] = path[str(state)] + [2]
                    try:    # [Year Constraints] ???? may need more considerations, narrow the constraint
                        tmp = int(tokens[t])
                        if tmp>= 1980 and tmp<=datetime.now().year:
                            new_path[str(i)] = path[str(state)] + [5]
                    except:
                        pass

            path = new_path

        # print 'Total log_V: ', sorted(log_V[-1].iteritems(), key=operator.itemgetter(1), reverse=True)
        # print 'Total path: ', path
        (prob, state) = max([(log_V[len(encoded_sequence)-1][str(j)], j) for j in xrange(self.state_dim)])
        candidate_range = [j for j in xrange(self.state_dim)]
        while 3 not in path[str(state)]:   #Cannot exclude title
            try:
                candidate_range.remove(state)
            except:
                pass

            if len(candidate_range) == 0:   #???? Title not in any path
                (prob, state) = max([(log_V[len(encoded_sequence)-1][str(j)], j) for j in xrange(self.state_dim)])
                break
            (prob, state) = max([(log_V[len(encoded_sequence)-1][str(j)], j) for j in candidate_range])
        
        return path[str(state)]


    # TO_DO: For normalizing probability to sequence length
    def z_score(self, x):
        pass



if __name__ == '__main__':
    # start_time = time.time()

    # observation_sequences_entire1, label_sequences_entire1 = get_training_samples('http://scholar.google.com/citations?user=Kv9AbjMAAAAJ&hl=en')      # large training sample
    # observation_sequences_entire2, label_sequences_entire2 = get_training_samples('http://scholar.google.com/citations?user=YU-baPIAAAAJ&hl=en')    # small training sample
    # observation_sequences = observation_sequences_entire1 + observation_sequences_entire2
    # label_sequences = label_sequences_entire1 + label_sequences_entire2

    
    # observation_sequences, label_sequences = get_training_samples_raw_cora()
    
    # Serialize training data
    # fp1 = open('observation_sequences.pkl', 'wb')
    # fp2 = open('label_sequences.pkl', 'wb')
    # pickle.dump(observation_sequences, fp1)
    # pickle.dump(label_sequences, fp2)
    # fp1.close()
    # fp2.close()

    # training_generation_time = time.time()
    # print 'STATS: Training Generation takes %d seconds!' % (training_generation_time - start_time)
    # Reading data from pickle
    observation_sequences = pickle.load(open('observation_sequences.pkl', 'rb'))
    label_sequences = pickle.load(open('label_sequences.pkl', 'rb'))

    log_err('Training sample retrieved!')
    # hmm = HMM('cora', 13)
    hmm = HMM('old', 6)
    log_err('Start Training...')
    hmm.train(observation_sequences, label_sequences)
    hmm.print_model()
    hmm.decode("A. Cau, R. Kuiper, and W.-P. de Roever. Formalising Dijkstra's development strategy within Stark's formalism. In C. B. Jones, R. C. Shaw, and T. Denvir, editors, Proc. 5th. BCS-FACS Refinement Workshop at San Francisco, 1992.")
    hmm.decode("Huizhong Duan, Emre Kiciman, ChengXiang Zhai,  Click Patterns: An Empirical Representation of Complex Query Intents  ,    Proceedings of the 21st ACM International Conference on Information and Knowledge Management (CIKM'12), to appear.")
    hmm.decode("Yue Lu, Hongning Wang, ChengXiang Zhai, Dan Roth,   Unsupervised Discovery of Opposing Opinion Networks From Forum Discussions,    Proceedings of the 21st ACM International Conference on Information and Knowledge Management (CIKM'12), to appear.")
    hmm.decode("Bin Tan, Yuanhua Lv, ChengXiang Zhai,  Mining long-lasting exploratory user interests from search history,    Proceedings of the 21st ACM International Conference on Information and Knowledge Management (CIKM'12), to appear.")
    hmm.decode("Yuanhua Lv, ChengXiang Zhai, Query Likelihood with Negative Query Generation,    Proceedings of the 21st ACM International Conference on Information and Knowledge Management (CIKM'12), to appear.")
    hmm.decode("V.G.Vinod Vydiswaran, ChengXiang Zhai, Dan Roth, and Peter Pirolli, BiasTrust: Teaching biased users about controversial topic,    Proceedings of the 21st ACM International Conference on Information and Knowledge Management (CIKM'12), to appear.")
    hmm.decode("Huizhong Duan, Yanen Li, ChengXiang Zhai and Dan Roth, A Discriminative Model for Query Spelling Correction with Latent Structural SVM,  Proceedings of EMNLP-CoNLL  2012 (EMNLP'12), pages 1511-1521, 2012.")
    hmm.decode("Yanen Li, Huizhong Duan, ChengXiang Zhai,  A Generalized Hidden Markov Model with Discriminative Training for Query Spelling Correction ,  Proceedings of ACM SIGIR 2012 (SIGIR'12), pages 611-620, 2012. ")
    hmm.decode("Parikshit Sondhi, Jimeng Sun, Hanghang Tong, ChengXiang Zhai, SympGraph: A Mining Framework of Clinical Notes through Symptom Relation Graphs, Proceedingsof  KDD 2012 (KDD'12), pages 1167-1175, 2012.")
    hmm.decode("Parikshit Sondhi, Jimeng Sun, ChengXiang Zhai, Robert Sorrentino and Martin S. Kohn,  Leveraging Medical Thesauri and Physician Feedback for Improving Medical Literature Retrieval for Case Queries, Journal of American Medical Informatics Association (JAMIA), 19(5): 851-858 (2012).")
    hmm.decode("Kavita Ganesan, Chengxiang Zhai and Evelyne Viegas,  Micropinion Generation: An Unsupervised Approach to Generating Ultra-Concise Summaries of Opinions,  Proceedings of the World Wide Conference 2012 ( WWW'12), pages 869-878, 2012. (acceptance rate 12%) ")
    hmm.decode("Alex Kotov, ChengXiang Zhai, Tapping into Knowledge Base for Concept Feedback: Leveraging ConceptNet to Improve Search Results for Difficult Queries, Proceedings of the 5th ACM International Conference on Web Search and Data Mining (WSDM'12), pages 403-412, 2012. (acceptance rate 21%)")
    hmm.decode("Shima Gerani, ChengXiang Zhai, Fabio Crestani, Score Transformation in Linear Combination for Multi-Criteria Relevance Ranking , Proceedings of the 34th European Conference on Information Retrieval (ECIR'12), pages 256-267, 2012. (acceptance rate 21%)")
    hmm.decode("Parikshit Sondhi, V.G.Vinod Vydiswaran, ChengXiang Zhai, Reliability Prediction of Webpages in the Medical Domain, Proceedings of the 34th European Conference on Information Retrieval (ECIR'12), pages 219-231, 2012.(acceptance rate 21%)")
    hmm.decode("Maryam Karimzadehgan, Chengxiang Zhai, Axiomatic Analysis of Translation Language Model For Information Retrieval , Proceedings of the 34th European Conference on Information Retrieval (ECIR'12), pages 268-280, 2012.  (acceptance rate 21%)")
    hmm.decode("Yuanhua Lv, Chengxiang Zhai, A Log-logistic Model-based Interpretation of TF Normalization of BM25, Proceedings of the 34th European Conference on Information Retrieval (ECIR'12), pages 244-255, 2012.  (acceptance rate 21%)")
    hmm.decode("Kavita Ganesan, ChengXiang Zhai, Opinion-based Entity Ranking, Information Retrieval, 15(2): 116-150 (2012)")
    hmm.decode("Duo Zhang, ChengXiang Zhai, Jiawei Han, MiTexCube: MicroTextCluster Cube for Online Analysis of Text Cells,  Proceedings of NASA Conference on Intelligent Data Understanding 2011, to appear.")
    hmm.decode("Alexander Kotov, ChengXiang Zhai, An Exploration of the Potential Effectiveness of Interactive Sense Feedback for Difficult Queries,  Proceedings of the 20th ACM International Conference on Information and Knowledge Management (CIKM'11), pages 163-172, 2011.")
    hmm.decode("Yuanhua Lv, ChengXiang Zhai, Lower Bounding Term Frequency Normalization,  Proceedings of the 20th ACM International Conference on Information and Knowledge Management (CIKM'11), pages 7-16, 2011. ( Best Student Paper Award) ")
    hmm.decode("Huizhong Duan, Rui Li, ChengXiang Zhai, Automatic Query Reformulation with Syntactic Operators to Alleviate Search Difficulty,  Proceedings of the 20th ACM International Conference on Information and Knowledge Management (CIKM'11), poster paper, pages 2037-2040, 2011.")
    hmm.decode("Yuanhua Lv, ChengXiang Zhai, Adaptive Term Frequency Normalization for BM25, Proceedings of the 20th ACM International Conference on Information and Knowledge Management (CIKM'11), poster paper, pages 1985-1988, 2011.")
    hmm.decode("Maryam Karimzadehgan, ChengXiang Zhai,Improving Retrieval Accuracy of Difficult Queries through Generalizing Negative Document Language Models, Proceedings of the 20th ACM International Conference on Information and Knowledge Management (CIKM'11), pages 27-36, 2011.")
    hmm.decode("Hongning Wang, Yue Lu, ChengXiang Zhai,  Latent Aspect Rating Analysis without Aspect Keyword Supervision, Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'11), 2011, pages 618-626. ( 17.5%% acceptance)")
    hmm.decode("V.G.Vinod Vydiswaran, ChengXiang Zhai, Dan Roth,Content-driven Trust Propagation Framework ,  Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'11), 2011, pages 974-982. ( 17.5%% acceptance)")
    hmm.decode("Hongning Wang, Chi Wang, ChengXiang Zhai, Jiawei Han, Learning Online Discussion Structures by Conditional Random Fields, Proceedings of the 34th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'11 ), 2011, pages 435-444. ( 20%% acceptance")
    hmm.decode("Yanen Li, Bo-June Hsu, ChengXiang Zhai, Kuansan Wang,  Unsupervised Query Segmentation Using Clickthrough for Information Retrieval,  Proceedings of the 34th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'11 ), 2011, pages 285-294. ( 20%% acceptance)")
    hmm.decode("Yuanhua Lv, ChengXiang Zhai, Wan Chen,   A Boosting Approach to Improving Pseudo-Relevance Feedback,  Proceedings of the 34th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'11 ), 2011, pages 165-174. ( 20%% acceptance)")
    hmm.decode("Hongning Wang, Duo Zhang, ChengXiang Zhai,  Structural Topic Model for Latent Topical Structure Analysis, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (ACL HTL'11), to appear.")
    hmm.decode("Yue Lu, Malu Castellanos, Umeshwar Dayal, ChengXiang Zhai, Automatic Construction of a Context-Aware Sentiment Lexicon: An Optimization Approach,   Proceedings of the World Wide Conference 2011 ( WWW'11), pages 347-356.")
    hmm.decode("Zhijun Yin, Liangliang Cao, Jiawei Han, Chengxiang Zhai, and Thomas Huang, Geographical Topic Discovery and Comparison,    Proceedings of the World Wide Conference 2011 ( WWW'11), pages 247-256.")
    hmm.decode("Huizhong Duan and Chengxiang Zhai,  Exploiting Thread Structure to Improve Smoothing of Language Models for Forum Post Retrieval, Proceedings of the 33rd European Conference on Information Retrieval (ECIR'11), to appear.")
    hmm.decode("Alex Kotov, ChengXiang Zhai, Richard Sproat, Mining Named Entities with Temporally Correlated Bursts from Multilingual Web News Streams,  Proceedings of WSDM 2011, to appear.")
    hmm.decode("Hui Fang, Tao Tao, ChengXiang Zhai,  Diagnostic Evaluation of Information RetrievalModels,  ACM Transactions on Information Systems (ACM TOIS), to appear.")
    hmm.decode("Yue Lu, Qiaozhu Mei, ChengXiang Zhai.  Investigating Task Performance of Probabilistic Topic Models - An Empirical Study of PLSA and LDA, Information Retrieval, vol. 14, no. 2, April, 2011.")
    hmm.decode("Yanen Li, Jia Hu, ChengXiang Zhai, Ye Chen. Improving One-Class Collaborative Filtering by Incorporating Rich User Information, Proceedings of the 19th ACM International Conference on Information and Knowledge Management (CIKM'10), pages 959-968, 2010. ( 13.4%% acceptance)")
    hmm.decode("Michael J. Paul, ChengXiang Zhai and Roxana Girju. Summarizing Contrastive Viewpoints In Opinionated Text,  Proceedings of the 2010 Conference on Empirical Methods in Natural Language Processing (EMNLP'10), pages 65-75, 2010. ( 25%% acceptance)")
    hmm.decode("Kavita Ganesan, ChengXiang Zhai, Jiawei Han. Opinosis: A Graph Based Approach to Abstractive Summarization of Highly Redundant Opinions,  Proceedings of COLING 2010, pages 340-348.")
    hmm.decode("Yue Lu, Huizhong Duan, Hongning Wang and ChengXiang Zhai. Exploiting Structured Ontology to Organize Scattered Online Opinions,  Proceedings of COLING 2010, pages 734-742.")
    hmm.decode("Parikshit Sondhi, Manish Gupta, ChengXiang Zhai and Julia Hockenmaier.Shallow Information Extraction from Medical Forum Data,   Proceedings of COLING 2010, pages 1158-1166.")
    hmm.decode("Hongning Wang, Yue Lu, ChengXiang Zhai.  Latent Aspect Rating Analysis on Review Text Data: A Rating Regression Approach,  Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'10), pages 115-124, 2010.")
    hmm.decode("Xin He, Yanen Li, Radhika Khetani, Barry Sanders, Yue Lu, Xu Ling, ChengXiang Zhai, Bruce Schatz. BSQA:  integrated text mining using entity relation semantics extracted from biological literature of insects,  Nucleic Acids Research . download")
    hmm.decode("Xin He, Moushumi Sen Sarma, Xu Ling, Brant Chee, ChengXiang Zhai, Bruce Schatz. Identifying overrepresented concepts in gene lists fromliterature: a statistical approach based on Poisson mixturemodel,  BMC Bioinformatics 2010, 11:272 (20 May 2010). download")
    hmm.decode("Duo Zhang,  Qiaozhu Mei, ChengXiang Zhai.  Cross-Lingual Latent Topic Extraction, Proceedings of the 48th Annual Meeting of the Association forComputational Linguistics ( ACL'10), pages 1128-1137, 2010.")
    hmm.decode("Maryam Karimzadehgan, ChengXiang Zhai,  Estimation of Statistical Translation Models Based on Mutual Information for Ad Hoc Information Retrieval ,   Proceedings of the 33rd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'10 ), pages 323-330, 2010. ( 16.7%% acceptance)")
    hmm.decode("Yuanhua Lv, ChengXiang Zhai,   Positional Relevance Model for Pseudo-Relevance Feedback ,   Proceedings of the 33rd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'10 ), pages 579-586, 2010. ( 16.7%% acceptance)")
    hmm.decode("Alexander Kotov, ChengXiang Zhai,  Towards Natural Question-Guided Search,   Proceedings of the World Wide Conference 2010 ( WWW'10), pages 541-550.")
    hmm.decode("Hyun Duk Kim, ChengXiang Zhai, Jiawei Han,  Aggregation of Multiple Judgments forEvaluating Ordered Lists, Proceedings of the 32nd European Conference on Information Retrieval (ECIR'10), pages 166-178, 2010. (22%% acceptance)")
    hmm.decode("Xuanhui Wang, Bin Tan, Azadeh Shakery, ChengXiang Zhai,   Beyond Hyperlinks: Organizing Information Footprints in Search Logs to Support Effective Browsing, Proceedings of the 18th ACM International Conference on Information and Knowledge Management  ( CIKM'09), pages 1237-1246, 2009. ( full paper, 14.5%% acceptance)  ")
    hmm.decode("Hyun Duk Kim, ChengXiang Zhai, Generating Comparative Summaries of Contradictory Opinions in Text,  Proceedings of the 18th ACM International Conference on Information and Knowledge Management  ( CIKM'09), pages 385-394, 2009. ( full paper, 14.5%% acceptance) ")
    hmm.decode("Yuanhua Lv, ChengXiang Zhai,  Adaptive Relevance Feedback in Information Retrieval,  Proceedings of the 18th ACM International Conference on Information and Knowledge Management  ( CIKM'09), pages 255-264, 2009. ( full paper, 14.5%% acceptance) ")
    hmm.decode("Yuanhua Lv, ChengXiang Zhai,  Positonal Language Models for Information Retrieval, Proceedings of the 32nd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'09 ), pages 299-306, 2009.  ( 16%% acceptance) ")
    hmm.decode("Younhee Ko, ChengXiang Zhai, Sandra Rodriguez-Zas,  Inference of Gene Pathways using Mixture Bayesian Networks, BMC Systems Biology, 3:54, 2009, doi:10.1186/1752-0509-3-54. ")
    hmm.decode("Duo Zhang, ChengXiang Zhai, Jiawei Han,  Topic Cube: Topic Modeling for OLAP on Multidimensional Text Databases, Proceedings of 2009 SIAM International Conference on Data Mining (SDM'09), pages 1123-1134, 2009. ( 16%% acceptance")
    hmm.decode("Yue Lu, ChengXiang Zhai, Neel Sundaresan, Rated Aspect Summarization of Short Comments,   Proceedings of the World Wide Conference 2009 ( WWW'09), pages 131-140. ( 12%% acceptance)")
    hmm.decode("Yue Lu, Hui Fang, ChengXiang Zhai, An Empirical Study of Gene Synonym Query Expansion in Biomedical Information Retrieval,  Information Retrieval, Volume 12, Number 1, Feb.  2009, Pages 51-68.  link")
    hmm.decode("ChengXiang Zhai, Statistical Language Models for Information Retrieval: A Critical Review,  Foundations and Trends in Information Retrieval, Vol. 2, No. 3 (2008), pages 137-215, doi:10.1561/1500000008.")
    hmm.decode("ChengXiang Zhai,  Statistical Language Models for Information Retrieval (Synthesis Lectures Series on Human Language Technologies), Morgan & Claypool Publishers, 2008.")
    hmm.decode("Bo Jin, Brian Muller, ChengXiang Zhai, Xinghua Lu,  Multi-label literature classification based on the Gene Ontology graph, BMC Bioinformatics, 2008, 9:525, doi:10.1186/1471-2105-9-525.")
    hmm.decode("Maryam Karimzadehgan, ChengXiang Zhai, Geneva Belford,  Multi-Aspect Expertise Matchingfor Review Assignment, Proceedings of the 17th ACM International Conference on Information and Knowledge Management  ( CIKM'08), pages 1113-1122.  (17%% acceptance)")
    hmm.decode("Xuanhui Wang, ChengXiang Zhai, Mining term association patterns from search logs for effective query reformulation, Proceedings of the 17th ACM International Conference on Information and Knowledge Management  ( CIKM'08), pages 479-488. (17%% acceptance)")
    hmm.decode("Deng Cai, Qiaozhu Mei, Jiawei Han, ChengXiang Zhai,  Modeling Hidden Topics on Document Manifold ,  Proceedings of the 17th ACM International Conference on Information and Knowledge Management  ( CIKM'08), pages 911-920.  (17%% acceptance)")
    hmm.decode("Xu Ling, Qiaozhu Mei, ChengXiang Zhai, Bruce R. Schatz,  Mining multi-faceted overviews of arbitrary topics in a text collection, Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'08), pages 497-505, 2008. ( 20%% acceptance)")
    hmm.decode("Qiaozhu Mei, Duo Zhang, ChengXiang Zhai. Smoothing Language Models with Document and Word Graphs , Proceedings of the 31st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'08 ), pages 611-618.  ( 17%% acceptance)")
    hmm.decode("Xuanhui Wang, Hui Fang, ChengXiang Zhai.  A study of methods for  negative relevance feedback , Proceedings of the 31st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'08 ), pages 219-226. ( 17%% acceptance)")
    hmm.decode("Qiaozhu Mei, ChengXiang Zhai.  Generating Impact-Based Summaries for Scientific Literature , Proceedings of the 46th Annual Meeting of the Association forComputational Linguistics: Human Language Technologies ( ACL-08:HLT), pages 816-824. (25%% acceptance)")
    hmm.decode("Yue Lu, ChengXiang Zhai.Opinion Integration Through Semi-supervised TopicModeling,   Proceedings of the World Wide Conference 2008 ( WWW'08), pages 121-130. ( 12%% acceptance)")
    hmm.decode("Qiaozhu Mei, Deng Cai, Duo Zhang, ChengXiang Zhai. Topic Modeling with Network Regularization,   Proceedings of the World Wide Conference 2008 ( WWW'08), pages 101-110. (12%% acceptance)")
    hmm.decode("Azadeh Shakery, ChengXiang Zhai.Smoothing Document Language Models with ProbabilisticTerm Count Propagation,  Information Retrieval, 11(2), 2008, pages 139-164.")
    hmm.decode("Xuanhui Wang, Tao Tao, Jian-Tao Sun, Azadeh Shakery, and ChengXiang Zhai, DirichletRank:Solving the Zero-One Gap Problem of PageRank, ACM Transactions on Information Systems,  26(2), 2008, Article No. 10.")
    hmm.decode("Qiaozhu Mei, Dong Xin, Hong Cheng, Jiawei Han, and ChengXiang Zhai,  Semantic Annotation of Frequent Patterns, ACM Transactions on Knowledge Discovery from Data,  1(3), Dec. 2007, Article No. 11.")
    hmm.decode("Jing Jiang, ChengXiang Zhai,  A Two-Stage Approach to Domain Adaptation for Statistical Classifiers ,  Proceedings of the 16th ACM International Conference on Information and Knowledge Management  ( CIKM'07), pages 401-410. ( full paper, 17%% acceptance)")
    hmm.decode("Xuanhui Wang, Hui Fang, ChengXiang Zhai,  Improve Retrieval Accuracy for Difficult Queries using Negative Feedback ,  Proceedings of the 16th ACM International Conference on Information and Knowledge Management  ( CIKM'07), pages 991-994. ( short paper, 26%% acceptance)")
    hmm.decode("Shui-Lung Chuang, Kevin Chen-Chuan Chuang, and ChengXiang Zhai, Context-Aware Wrapping: Synchronized Data Extraction, Proceedings of the 33rd Very Large Data Bases Conference (VLDB'07),pages 699-710. (17.5%% acceptance)")
    hmm.decode("Xuehua Shen, Bin Tan, and ChengXiang Zhai,  Privacy Protection in Personalized Search, ACM SIGIR Forum , 41(1), pages 4-17.")
    hmm.decode("Qiaozhu Mei, Xuehua Shen, and ChengXiang Zhai,  Automatic Labeling of Multinomial Topic Models , Proceedings of the 2007 ACM SIGKDD  International Conference on Knowledge Discovery and Data Mining  (KDD'07 ), pages 490-499. ( 19%% acceptance )")
    hmm.decode("Xuanhui Wang, ChengXiang Zhai, Xiao Hu, and Richard Sproat,    Mining Correlated Bursty Topic Patterns from Coordinated Text Streams , Proceedings of the 2007 ACM SIGKDD  International Conference on Knowledge Discovery and Data Mining  (KDD'07 ), pages 784-793. (19%% acceptance rate) ")
    hmm.decode("Xuanhui Wang, ChengXiang Zhai,  Learn from Web Search Logs to Organize Search Results,  Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'07 ), pages 87-94. ( 18%% acceptance) ")
    hmm.decode("Bin Tan, Atulya Velivelli, Hui Fang, ChengXiang Zhai, Term Feedback for Information Retrieval with Language Models,  Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'07 ), pages 263-270. ( 18%% acceptance)")
    hmm.decode("Qiaozhu Mei, Hui Fang, ChengXiang Zhai,  A Study of Poisson Query Generation Model for Information Retrieval, Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'07 ), pages 319-326. ( 18%% acceptance) ")
    hmm.decode("Tao Tao, ChengXiang Zhai,  An Exploration of Proximity Measures in Information Retrieval,  Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'07 ), pages 295-302. ( 18%% acceptance)")
    hmm.decode("Jing Jiang and ChengXiang Zhai,  An Empirical Study of Tokenization Strategies for Biomedical Information  Retrieval,  Information Retrieval, 10(4-5), Oct. 2007, pp. 341-363.")
    hmm.decode("Jing Jiang and ChengXiang Zhai,  Instance Weighting for Domain Adaptation in NLP, Proceedings of ACL 2007, pages 264-271.")
    hmm.decode("Qiaozhu Mei, Xu Ling, Matthew Wondra, Hang Su, ChengXiang Zhai,  Topic Sentiment Mixture: Modeling Facets and Opinions in Weblogs,  Proceedings of the World Wide Conference 2007 ( WWW'07), pages 171-180.")
    hmm.decode("Hui Fang, ChengXiang Zhai,  Probabilistic Models for Expert Finding , Proceedings of the 29th European Conference on Information Retrieval (ECIR'07), pages 418-430. ( 19%% acceptance)")
    hmm.decode("Jing Jiang, ChengXiang Zhai,   A Systematic Exploration of The Feature Space for           Relation Extraction, Proceedings of Human Language Technologies: The Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT 2007), pages 113-120. ( 24%% acceptance)")
    hmm.decode("Xu Ling, Jing Jiang, Xin He, Qiaozhu Mei, ChengXiang Zhai, Bruce Schatz, Generating Semi-Structured Gene Summaries from Biomedical Literature, Information Processing and Management, 43(6), Nov. 2007, pp. 1777-1791.")
    hmm.decode("Saurabh Sinha, Xu Ling, Charles W. Whitfield, ChengXiang Zhai, and Gene E. Robinson, Genome scan for cis-regulatory DNA motifs associated with social behavior in honey bees ,  Proceedings of National Academy of Sciences of the United States of America (PNAS) ,103(44), Oct. 2006, pages 16352-16357. URL")
    hmm.decode("Jing Jiang and ChengXiang Zhai, Extraction of coherent relevant passagesusing hidden Markov models,  ACM Transactions on InformationSystems, 24(3), July 2006, pages 295-319. URL")
    hmm.decode("Azadeh Shakery and ChengXiang Zhai, A probabilistic relevance propagation model for hypertext retrieval,In  Proceedings of the 15th ACM International Conference on Information and Knowledge Management  ( CIKM'06), pages 550-558. ( 15%% acceptance)")
    hmm.decode("Rong Jin, Luo Si, and ChengXiang Zhai,  A study of mixture models for collaborative filtering, Information Retrieval,9(3), Jun. 2006, pages 357-382.  URL")
    hmm.decode("Bin Tan, Xuehua Shen, ChengXiang Zhai, Mining long-term search history to improve search accuracy , Proceedings of the 2006 ACM SIGKDD  International Conference on Knowledge Discovery and Data Mining , (KDD'06 ), pages 718-723. (poster paper, 23%% acceptance) ")
    hmm.decode("Qiaozhu Mei, ChengXiang Zhai,   A Mixture Model for Contextual Text Mining, Proceedings of the 2006 ACM SIGKDD  International Conference on Knowledge Discovery and Data Mining , (KDD'06 ), pages 649-655. (poster paper, 23%% acceptance)")
    hmm.decode("Qiaozhu Mei, Dong Xin, Hong Cheng, Jiawei Han, ChengXiang Zhai,   Generating Semantic Annotations for Frequent Patterns with Context Analysis ,  Proceedings of the 2006 ACM SIGKDD  International Conference on Knowledge Discovery and Data Mining , (KDD'06 ), pages 337-346.   Best Student Paper Award Runner-Up. (full paper, 11%% acceptance)")
    hmm.decode("Tao Tao, Su-Youn Yoon, Andrew Fister, Richard Sproat and ChengXiang Zhai, Unsupervised Named Entity Transliteration Using Temporal and Phonetic Correlation , Proceedings of the 2006 Conference on Empirical Methods in Natural Language Processing  (EMNLP 2006), pages 250-257. ( 31%% acceptance)")
    hmm.decode("Richard Sproat, Tao Tao and ChengXiang Zhai, Named Entity Transliteration with Comparable Corpora, Proceedings of COLING-ACL 2006, pages 73-80.  ( 23%% acceptance)")
    hmm.decode("Xuanhui Wang, Jian-Tao Sun, Zheng Chen, ChengXiang Zhai,Latent Semantic Analysis for Multiple-Type Interrelated Data Objects  Proceedings of the 29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'06 ), pages 236-243. ( 19%% acceptance) ")
    hmm.decode("Hui Fang, ChengXiang Zhai,Semantic Term Matching in Axiomatic Approaches to Information Retrieval  Proceedings of the 29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'06 ), pages 115-122. ( 19%% acceptance)")
    hmm.decode("Tao Tao, ChengXiang Zhai,Regularized Estimation of Mixture Models for Robust Pseudo-Relevance Feedback Proceedings of the 29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'06 ), pages 162-169. ( 19%% acceptance)")
    hmm.decode("Jing Jiang, ChengXiang Zhai,Exploiting Domain Structure for Named Entity Recognition. Proceedings of HLT/NAACL 2006, pages 74-81. ( 25%% acceptance)")
    hmm.decode("Tao Tao, Xuanhui Wang, Qiaozhu Mei, ChengXiang Zhai,Language Model Information Retrieval with Document Expansion. Proceedings of  HLT/NAACL 2006, pages 407-414. ( 25%% acceptance)")
    hmm.decode("Qiaozhu Mei, Chao Liu, Hang Su, and ChengXiang Zhai, A Probabilistic Approach to Spatiotemporal Theme Pattern Mining on Weblogs.   Proceedings of the World Wide Web Conference 2006  ( WWW'06), pages 533-542.  (11%% acceptance)")
    hmm.decode("Xu Ling, Jing Jiang, Xin He, Qiaozhu Mei, ChengXiang Zhai, and Bruce Schatz,  Automatically Generating Gene Summaries from Biomedical Literature . In  Proceedings of Pacific Symposium on Biocomputing 2006 (PSB'06), pages 40-51")
    hmm.decode("ChengXiang Zhai and John Lafferty,  A risk minimization framework for information retrieval ,  Information Processing and Management ( IP &M ), 42(1), Jan. 2006. pages 31-55.  URL")
    hmm.decode("Xuehua Shen, Bin Tan, and ChengXiang Zhai,  Implicit User Modeling for Personalized Search ,In  Proceedings of the 14th ACM International Conference on Information and Knowledge Management  ( CIKM'05), pages 824-831. ")
    hmm.decode("Qiaozhu Mei, ChengXiang Zhai,  Discovering Evolutionary Theme Patterns from Text -- An Exploration of Temporal Text Mining,  Proceedings of the 2005 ACM SIGKDD  International Conference on Knowledge Discovery and Data Mining , (KDD'05 ), pages 198-207, 2005.")
    hmm.decode("Tao Tao, ChengXiang Zhai,  Mining Comparable Bilingual Text Corpora for Cross-Language Information Integration , Proceedings of the 2005 ACM SIGKDD International Conference on Knowledge Discovery and Data Mining  (KDD'05 ), pages 691-696, 2005. ")
    hmm.decode("Hui Fang, ChengXiang Zhai,  An Exploration of Axiomatic Approach to Information Retrieval ,   Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'05 ), 480-487, 2005")
    hmm.decode("Xuehua Shen, ChengXiang Zhai,  Active Feedback in Ad Hoc Information Retrieval,   Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'05), 59-66, 2005")
    hmm.decode("Xuehua Shen, Bin Tan, ChengXiang Zhai,  Context-Sensitive Information Retrieval with Implicit Feedback,   Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'05), 43-50, 2005.  ")
    hmm.decode("Tao Tao, ChengXiang Zhai, Xinghua Lu, and Hui Fang, A study of statistical methods for function prediction of protein motifs , Applied Bioinformatics, Volume 3, No. 2-3, pages 115-124.  (BLM 03 paper: ps,")
    hmm.decode("Xinghua Lu, Chengxiang Zhai , Vanathi Gopalakrishnan,  and Bruce G Buchanan,  Automatic annotation of protein motif function with Gene Ontology terms, BMC Bioinformatics 2004, 5:122. (url) (Impact factor=5.42, as of 2006)")
    hmm.decode("Hui Fang, Tao Tao, ChengXiang Zhai, A formal study of information retrieval heuristics,  Proceedings of the 27th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'04), pages 49-56, 2004.  Best Paper Award.")
    hmm.decode("ChengXiang Zhai, Atulya Velivelli, Bei Yu, A cross-collection mixture model for comparative text mining,  Proceedings of ACM KDD 2004 ( KDD'04 ), pages 743-748, 2004. ")
    hmm.decode("Tao Tao, ChengXiang Zhai,  A Mixture Clustering Model for Pseudo Feedback in Information Retrieval , Proceedings of the 2004 Meeting of the International Federation of Classification Societies ( IFCS'04), pages 541-552.  Invited Paper.")
    hmm.decode("ChengXiang Zhai, John Lafferty,  A study of smoothing methods for language models applied to information retrieval , ACM Transactions on Information Systems ( ACM TOIS ), Vol. 22, No. 2, April 2004, pages 179-214. ( ps)")
    hmm.decode("Hwanjo Yu, ChengXiang Zhai, and Jiawei Han,  Text Classification from Positive and Unlabeled Documents , Proceedings of ACM CIKM 2003 (CIKM'03), pages 232-239, 2003.")
    hmm.decode("Jin Rong, Luo Si, ChengXiang Zhai, and Jamie Callan, Collaborative Filtering with Decoupled Models for Preferences and Ratings , Proceedings of ACM CIKM 2003  (CIKM'03 ), pages 301-316, 2003. ps,")
    hmm.decode("ChengXiang Zhai, William W. Cohen, and John Lafferty,  Beyond Independent Relevance: Methods and Evaluation Metrics for Subtopic Retrieval ,   Proceedings of the 26th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'03 ), pages 10-17, 2003. ps,")
    hmm.decode("Rong Jin, Luo Si, and ChengXiang Zhai,  Preference-based Graphic Models for Collaborative Filtering, In  Proceedings of UAI 2003 (UAI'03 ), pages 329-336, 2003. ps,")
    hmm.decode("John Lafferty and Chengxiang Zhai, Probabilistic relevance models based on document and query generation , In  Language Modeling and Information Retrieval, Kluwer International Series on Information Retrieval, Vol. 13, 2003.  ps")
    hmm.decode("ChengXiang Zhai,  Risk Minimization and Language Modeling in Information Retrieval, Ph.D. thesis, Carnegie Mellon University, 2002. (summary).")
    hmm.decode("ChengXiang Zhai and John Lafferty,  Two-Stage Language Models for Information Retrieval ,   Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'02), pages 49-56, 2002. ps,")
    hmm.decode("Rong Jin, Alex G. Hauptmann, and ChengXiang Zhai, Title LanguageModel for  Information Retrieval,   Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval ( SIGIR'02 ), pages 42-48, 2002.ps,")
    hmm.decode("Chengxiang Zhai and John Lafferty,  Model-based feedback in the language modeling approach to information retrieval ,  Proceedings of the Tenth ACM  International Conference on  Information and Knowledge Management (CIKM'01), pages 403-410, 2001. ps,")
    hmm.decode("Chengxiang Zhai and John Lafferty,   A study of smoothing methods forlanguage models applied to ad hoc information retrieval,Proceedings of the 24th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval  (SIGIR'01 ), pages 334-342, 2001. ps,")
    hmm.decode("John Lafferty and Chengxiang Zhai, Document language models, query models, and risk minimization for information retrieval , Proceedings of the 24th Annual International  ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'01 ), pages 111-119, 2001. ps")

    hmm.decode('Incremental and Accuracy-Aware Personalized PageRank through Scheduled Approximation. F. Zhu, Y. Fang, K. C.-C. Chang, and J. Ying. PVLDB, 6(6), 2013. In VLDB 2013.')
    hmm.decode('RoundTripRank: Graph-based Proximity with Importance and Specificity. Y. Fang, K. C.-C. Chang, and H. W. Lauw. In ICDE, 2013. Dataset')
    hmm.decode('Learning to Rank from Distant Supervision: Exploiting Noisy Redundancy for Relational Entity Search. M. Zhou, H. Wang, and K. C.-C. Chang. In ICDE, 2013.')
    hmm.decode('Multiple Location Profiling for Users and Relationships from Social Network and Content. R. Li, S. Wang, and K. C.-C. Chang. PVLDB, 5(11):1603-1614, 2012. In VLDB 2012.')
    hmm.decode('Towards Social User Profiling: Unified and Discriminative Influence Model for Inferring Home Locations. R. Li, S. Wang, H. Deng, R. Wang, and K. C.-C. Chang. In KDD, 2012.')
    hmm.decode('Confidence-Aware Graph Regularization with Heterogeneous Pairwise Features. Y. Fang, B.-J. P. Hsu, and K. C.-C. Chang. In SIGIR, 2012.')
    hmm.decode('Searching Patterns for Relation Extraction over the Web: Rediscovering the Pattern-Relation Duality. Y. Fang and K. C.-C. Chang. In WSDM, pages 825-834, 2011. (83/372=22%). BibTex')
    hmm.decode('Towards Rich Query Interpretation: Walking Back and Forth for Mining Query Templates. G. Agarwal, G. Kabra, and K. C.-C. Chang. In WWW, pages 1-10, 2010. (104/743=14%).')
    hmm.decode('Beyond Pages: Supporting Efficient, Scalable Entity Search. T. Cheng and K. C.-C. Chang. In EDBT, pages 15-26, 2010.')
    hmm.decode('Data-oriented Content Query System: Searching for Data into Text on the Web. M. Zhou, T. Cheng, and K. C.-C. Chang. In WSDM, pages 121-130, 2010. (45/290=15.5%).')
    hmm.decode('Integrating Web Query Results: Holistic Schema Matching. S.-L. Chuang and K. C.-C. Chang. In CIKM, pages 33-42, 2008. (132/772=17%).')
    hmm.decode('EntityRank: Searching Entities Directly and Holistically. T. Cheng, X. Yan, and K. C.-C. Chang. In Proceedings of the 33rd Very Large Data Bases Conference (VLDB 2007), pages 387-398, Vienna, Austria, September 2007. (91/538=16.9%).')
    hmm.decode('Context-Aware Wrapping: Synchronized Data Extraction. S.-L. Chuang, K. C.-C. Chang, and C. Zhai. In Proceedings of the 33rd Very Large Data Bases Conference (VLDB 2007), pages 699-710, Vienna, Austria, September 2007. (91/538=16.9%).')
    hmm.decode('Supporting Ranking and Clustering as Generalized Order-By and Group-By. C. Li, M. Wang, L. Lim, H. Wang, and K. C.-C. Chang. In Proceedings of the 2007 ACM SIGMOD Conference (SIGMOD 2007), pages 127-138, Beijing, China, June 2007. (70/480=14.6%). Slides')
    hmm.decode('Progressive and Selective Merge: Computing Top-K with Ad-hoc Ranking Functions. D. Xin, J. Han, and K. C.-C. Chang. In Proceedings of the 2007 ACM SIGMOD Conference (SIGMOD 2007), pages 103-114, Beijing, China, June 2007. (70/480=14.6%). Slides')
    hmm.decode('Entity Search Engine: Towards Agile Best-Effort Information Integration over the Web. T. Cheng and K. C.-C. Chang. In Proceedings of the Third Conference on Innovative Data Systems Research (CIDR 2007), pages 108-113, Asilomar, Ca., January 2007. Extended System Demo Description.')
    hmm.decode('Top-k Query Processing in Uncertain Databases. M. A. Soliman, I. F. Ilyas, and K. C.-C. Chang. In Proceedings of the 23rd International Conference on Data Engineering (ICDE 2007), pages 896-905, Istanbul, Turkey, April 2007. (122/659=18%).')
    hmm.decode('Collaborative Wrapping: A Turbo Framework for Web Data Extraction. S.-L. Chiang, K. C.-C. Chang, and C. Zhai. In Proceedings of the 23rd International Conference on Data Engineering (ICDE 2007), pages 1261-1262, Istanbul, Turkey, April 2007. (Poster Paper; 182/659=27%).')
    hmm.decode('Supporting Ad-hoc Ranking Aggregates. C. Li, K. C.-C. Chang, and I. F. Ilyas. In Proceedings of the 2006 ACM SIGMOD Conference (SIGMOD 2006), pages 61-72, Chicago, June 2006. (58/446=13%). Slides')
    hmm.decode('Boolean + Ranking: Querying a Database by K-Constrained Optimization. Z. Zhang, S. Hwang, K. C.-C. Chang, M. Wang, C. Lang, and Y. Chang. In Proceedings of the 2006 ACM SIGMOD Conference (SIGMOD 2006), pages 359-370, Chicago, June 2006. (58/446=13%). Slides')
    hmm.decode('Light-weight Domain-based Form Assistant: Querying Web Databases On the Fly. Z. Zhang, B. He, and K. C.-C. Chang. In Proceedings of the 31st Very Large Data Bases Conference (VLDB 2005), pages 97-108, Trondheim, Norway, August 2005. (32/195=16%). Slides')
    hmm.decode('Making Holistic Schema Matching Robust: An Ensemble Approach. B. He and K. C.-C. Chang. In Proceedings of the 2005 ACM SIGKDD Conference (KDD 2005), pages 429-438, Chicago, Illinois, August 2005. (14/75=19%). Slides')
    hmm.decode('RankSQL: Query Algebra and Optimization for Relational Top-k Queries. C. Li, K. C.-C. Chang, I. F. Ilyas, and S. Song. In Proceedings of the 2005 ACM SIGMOD Conference (SIGMOD 2005), pages 131-142, Baltimore, Maryland, June 2005. (66/431=15%). Slides')
    hmm.decode('Toward Large Scale Integration: Building a MetaQuerier over Databases on the Web. K. C.-C. Chang, B. He, and Z. Zhang. In Proceedings of the Second Conference on Innovative Data Systems Research (CIDR 2005), pages 44-55, Asilomar, Ca., January 2005. (26/86=30%). Slides')
    hmm.decode('RankFP: A Framework for Supporting Rank Formulation and Processing. H. Yu, S. Hwang, and K. C.-C. Chang. In Proceedings of the 21st International Conference on Data Engineering (ICDE 2005), pages 514-515, Tokyo, Japan, April 2005. (Poster Paper; 100/521=19%). Slides')
    hmm.decode('Optimizing Access Cost for Top-k Queries over Web Sources: A Unified Cost-based Approach. S. Hwang and K. C.-C. Chang. In Proceedings of the 21st International Conference on Data Engineering (ICDE 2005), pages 188-189, Tokyo, Japan, April 2005. (Poster Paper; 100/521=19%). Slides')
    hmm.decode('Organizing Structured Web Sources by Query Schemas: A Clustering Approach. B. He, T. Tao, and K. C.-C. Chang. In Proceedings of the 13th Conference on Information and Knowledge Management (CIKM 2004), pages 22-31, Washington, D.C., November 2004. (59/303=19%). Slides')
    hmm.decode('Optimal Multimodal Fusion for Multimedia Data Analysis. Y. Wu, E. Y. Chang, K. C.-C. Chang, and J. R. Smith. In Proceedings of the 12th ACM International Conference on Multimedia (MM 2004), pages 572-579, New York, October 2004. (56/330=17%).')
    hmm.decode('Discovering Complex Matchings across Web Query Interfaces: A Correlation Mining Approach. B. He, K. C.-C. Chang, and J. Han. In Proceedings of the 2004 ACM SIGKDD Conference (KDD 2004), pages 148-157, Seattle, Wa., August 2004. (40/337=12%). Slides')
    hmm.decode('Understanding Web Query Interfaces: Best-Effort Parsing with Hidden Syntax. Z. Zhang, B. He, and K. C.-C. Chang. In Proceedings of the 2004 ACM SIGMOD Conference (SIGMOD 2004), pages 117-118, Paris, France, June 2004. (69/431=16%). Slides')
    hmm.decode('Statistical Schema Matching across Web Query Interfaces. B. He and K. C.-C. Chang. In Proceedings of the 2003 ACM SIGMOD Conference (SIGMOD 2003), pages 217-228, San Diego, California, June 2003. (52/342=15%). Slides')
    hmm.decode('Heterogeneous Learner for Web Page Classification. H. Yu, K. C.-C. Chang, and J. Han. In Proceedings of the 2002 IEEE International Conference on Data Mining (ICDM 2002), pages 538-545, Maebashi, Japan, December 2002. (72/369=20%).')
    hmm.decode('PEBL: Positive Example Based Learning for Web Page Classification Using SVM. H. Yu, J. Han, and K. C.-C. Chang. In Proceedings of the 2002 ACM SIGKDD Conference (KDD 2002), pages 239-248, Edmonton, Alberta, Canada, July 2002. (44/308=14%).')
    hmm.decode('Minimal Probing: Supporting Expensive Predicates for Top-k Queries. K. C.-C. Chang and S.-W. Hwang. In Proceedings of the 2002 ACM SIGMOD Conference (SIGMOD 2002), pages 346-357, Madison, Wisconsin, June 2002. 42/239=18%. Slides')
    hmm.decode('NBDL: A CIS Framework for NSDL. J. Futrelle, K. C.-C. Chang, and S.-S. Chen. In Proceedings of the First ACM/IEEE Joint Conference on Digital Libraries (JCDL 2001), pages 124-125, Roanoke, Virginia, June 2001.')
    hmm.decode('Approximate Query Translation Across Heterogeneous Information Sources. K. C.-C. Chang and H. Garcia-Molina. In Proceedings of the 26th VLDB Conference (VLDB 2000), pages 566-577, Cairo, Egypt, September 2000. (53/351=15%). Extended Version')
    hmm.decode('Mind Your Vocabulary: Query Mapping Across Heterogeneous Information Sources. K. C.-C. Chang and H. Garcia-Molina. In Proceedings of the 1999 ACM SIGMOD Conference (SIGMOD 1999), pages 335-346, Philadelphia, Pa., June 1999. (42/205=20%). PS Extended Version')
    hmm.decode('Conjunctive Constraint Mapping for Data Translation. K. C.-C. Chang and H. Garcia-Molina. In Proceedings of the 3rd ACM International Conference on Digital Libraries (DL 1998), pages 49-58, Pittsburgh, Pa., June 1998. PS')
    hmm.decode('An Extensible Constructor Tool for the Rapid, Interactive Design of Query Synthesizers. M. Baldonado, S. Katz, A. Paepcke, K. C.-C. Chang, H. Garcia-Molina, and T. Winograd. In Proceedings of the 3rd ACM International Conference on Digital Libraries (DL 1998), pages 19-28, Pittsburgh, Pa., June 1998. PS')
    hmm.decode('PowerBookmarks: An Advanced Web Bookmark Database System and its Information Sharing and Management. W.-S. Li, Y.-L. Wu, C. Bufi, K. C.-C. Chang, D. Agrawal, and Y. Hara. In Proceedings of the 5th International Conference of Foundations of Data Organization (FODO 1998), Kobe, Japan, November 1998.')
    hmm.decode('Evaluating the Cost of Boolean Query Mapping. K. C.-C. Chang and H. Garcia-Molina. In Proceedings of the Second ACM International Conference on Digital Libraries (DL 1997), pages 103-112, Philadelphia, Pa., July 1997. (28/104=27%). PS')
    hmm.decode('Metadata for Digital Libraries: Architecture and Design Rationale. M. Baldonado, K. C.-C. Chang, L. Gravano, and A. Paepcke. In Proceedings of the Second ACM International Conference on Digital Libraries (DL 1997), pages 47-56, Philadelphia, Pa., July 1997. (28/104=27%). PS')
    hmm.decode('STARTS: Stanford Proposal for Internet Meta-Searching. L. Gravano, K. C.-C. Chang, H. Garcia-Molina, and A. Paepcke. In Proceedings of the 1997 ACM SIGMOD Conference (SIGMOD 1997), pages 207-218, Tucson, Ariz., May 1997. (42/202=21%). PS')
    hmm.decode('Probabilistic top-k and ranking-aggregate queries. M. A. Soliman, I. F. Ilyas, and K. C.-C. Chang. ACM Trans. Database Syst., 33(3), 2008.')
    hmm.decode('Trustworthy keyword search for compliance storage. S. Mitra, M. Winslett, W. W. Hsu, and K. C.-C. Chang. VLDB J., 17(2):225-242, 2008.')
    hmm.decode('Accessing the Deep Web: A Survey. B. He, M. Patel, Z. Zhang, and K. C.-C. Chang. Communications of the ACM, 50(5):94-101, May 2007.')
    hmm.decode('Optimizing Top-k Queries for Middleware Access: A Unified Cost-based Approach. S.-W. Hwang and K. C.-C. Chang. ACM Transactions on Database Systems (TODS), 32(1):5, March 2007.')
    hmm.decode('Probe Minimization by Schedule Optimization: Supporting Top-k Queries with Expensive Predicates. S.-W. Hwang and K. C.-C. Chang. IEEE Transactions on Knowledge and Data Engineering (TKDE), 19(5):646-662, May 2007.')
    hmm.decode('Automatic Complex Schema Matching across Web Query Interfaces: A Correlation Mining Approach. B. He and K. C.-C. Chang. ACM Transactions on Database Systems (TODS), 31(1):346-395, March 2006.')
    hmm.decode('Mining Semantics for Large Scale Integration on the Web: Evidences, Insights, and Challenges. K. C.-C. Chang, B. He, and Z. Zhang. SIGKDD Explorations, 6(2):67-76, December 2004.')
    hmm.decode('Editorial: Special Issue on Web Content Mining. B. Liu and K. C.-C. Chang. SIGKDD Explorations, 6(2):1-4, December 2004.')
    hmm.decode('A Holistic Paradigm for Large Scale Schema Matching. B. He and K. C.-C. Chang. SIGMOD Record, 33(4):20-25, December 2004. Invited paper.')
    hmm.decode('Structured Databases on the Web: Observations and Implications. K. C.-C. Chang, B. He, C. Li, M. Patel, and Z. Zhang. SIGMOD Record, 33(3):61-70, September 2004.')
    hmm.decode('PEBL: Web Page Classification without Negative Examples. H. Yu, J. Han, and K. C.-C. Chang. IEEE Transactions on Knowledge and Data Engineering, 16(1):70-81, January 2004. Special Section on Mining and Searching the Web.')
    hmm.decode('Data Mining for Web Intelligence. J. Han and K. C.-C. Chang. IEEE Computer, IEEE Computer Society, Washington, D.C., 35(11):64-70, November 2002.')
    hmm.decode('Approximate Query Mapping: Accounting for Translation Closeness. K. C.-C. Chang and H. Garcia-Molina. The VLDB Journal, VLDB Foundation, Saratoga, Calif., 10(2-3):155-181, September 2001. PS')
    hmm.decode('Using Distributed Objects to Build the Stanford Digital Library InfoBus. A. Paepcke, M. Baldonado, K. C.-C. Chang, S. Cousins, and H. Garcia-Molina. IEEE Computer, IEEE Computer Society, Washington, D.C., 32(2):80-87, February 1999.')
    hmm.decode('Predicate Rewriting for Translating Boolean Queries in a Heterogeneous Information System. K. C.-C. Chang, H. Garcia-Molina, and A. Paepcke. ACM Transactions on Information Systems, ACM Press, New York, 17(1):1-39, January 1999. PS')
    hmm.decode('Interoperability for Digital Libraries Worldwide. A. Paepcke, K. C.-C. Chang, H. Garcia-Molina, and T. Winograd. Communications of the ACM, ACM Press, New York, 41(4):33-43, April 1998. PS')
    hmm.decode('The Stanford Digital Library Metadata Architecture. M. Baldonado, K. C.-C. Chang, L. Gravano, and A. Paepcke. International Journal on Digital Libraries, Springer, Berlin, 1(2):108-121, September 1997. PS')
    hmm.decode('Boolean Query Mapping Across Heterogeneous Information Sources. K. C.-C. Chang, H. Garcia-Molina, and A. Paepcke. IEEE Transactions on Knowledge and Data Engineering, IEEE Computer Society, Washington, D.C., 8(4):515-521, August 1996. PS Extended Version')
    hmm.decode('Accessing the Web: From Search to Integration. K. C.-C. Chang and J. Cho. In Proceedings of the 2006 ACM SIGMOD Conference (SIGMOD 2006), pages 804-805, Chicago, June 2006. Tutorial description. Slides')
    hmm.decode('TEDAS: a Twitter Based Event Detection and Analysis System. R. Li, K. H. Lei, R. Khadiwala, and K. C.-C. Chang. In ICDE Conference, 2012. Demonstration description.')
    hmm.decode('DoCQS: a prototype system for supporting data-oriented content query. M. Zhou, T. Cheng, and K. C.-C. Chang. In SIGMOD Conference, pages 1211-1214, 2010. Demonstration description.')
    hmm.decode('AIDE: ad-hoc intents detection engine over query logs. Y. Jiang, H.-T. Yang, K. C.-C. Chang, and Y.-S. Chen. In SIGMOD Conference, pages 1091-1094, 2009. Demonstration description.')
    hmm.decode('Supporting Entity Search: a Large-Scale Prototype Search Engi1ne. T. Cheng, X. Yang, and K. C.-C. Chang. In Proceedings of the 2007 ACM SIGMOD Conference (SIGMOD 2007), pages 1144-1146, Beijing, China, June 2007. Demonstration description. (35/107 = 32%).')
    hmm.decode('URank: Top-k Query Processing for Uncertain Databases. M. Sliman, I. Ilyas, and K. C.-C. Chang. In Proceedings of the 2007 ACM SIGMOD Conference (SIGMOD 2007), pages 1082-1084, Beijing, China, June 2007. Demonstration description. (35/107 = 32%).')
    hmm.decode('Dewex: A Search Engine for Exploring the Deep Web. G. Kabra, Z. Zhang, and K. C.-C. Chang. In Proceedings of the 23rd International Conference on Data Engineering (ICDE 2007), pages 1511-1512, Istanbul, Turkey, April 2007. Demonstration description.')
    hmm.decode('RankSQL: Supporting Ranking Queries in Relational Database Management Systems. C. Li, M. A. Soliman, K. C.-C. Chang, and I. F. Ilyas. In Proceedings of the 31st Very Large Data Bases Conference (VLDB 2005), pages 1342-1345, Trondheim, Norway, August 2005. Demonstration description. (29/69 = 42%).')
    hmm.decode('MetaQuerier: Querying Structured Web Sources On-the-fly. B. He, Z. Zhang, and K. C.-C. Chang. In Proceedings of the 2005 ACM SIGMOD Conference (SIGMOD 2005), pages 927-929, Baltimore, Maryland, June 2005. Demonstration description. (24/71 = 34%).')
    hmm.decode('Towards Building a MetaQuerier: Extracting and Matching Web Query Interfaces. B. He, Z. Zhang, and K. C.-C. Chang. In Proceedings of the 21st International Conference on Data Engineering (ICDE 2005), pages 1098-1099, Tokyo, Japan, April 2005. Demonstration description.')
    hmm.decode('Towards Building a MetaQuerier: Extracting and Matching Web Query Interfaces. B. He, Z. Zhang, and K. C.-C. Chang. In NSF Information and Data Management (IDM) Workshop 2004, Boston, Massachussetts, October 2004. Demonstration description.')
    hmm.decode('Knocking the Door to the Deep Web: Integrating Web Query Interfaces. B. He, Z. Zhang, and K. C.-C. Chang. In Proceedings of the 2004 ACM SIGMOD Conference (SIGMOD 2004), pages 913-914, Paris, France, June 2004. Demonstration description.')
    hmm.decode('Knocking the Doors to the Deep Web: Understanding Web Query Interfaces. Z. Zhang, B. He, and K. C.-C. Chang. In NSF Information and Data Management (IDM) Workshop 2003, Seattle, Washington, September 2003. Demonstration description.')
    hmm.decode('PowerBookmarks: A System for Personalizable Web Information Organization, Sharing, and Management. W.-S. Li, K. C.-C. Chang, D. Agrawal, and et al. In Proceedings of the 1999 ACM SIGMOD Conference (SIGMOD 1999), pages 565-567, Philadelphia, Pa., June 1999. Demonstration description.')
    hmm.decode('Object Search: Supporting Structured Queries in Web Search Engines. K. Pham, N. Rizzolo, K. Small, K. C.-C. Chang, and D. Roth. In NAACL-HLT Workshop on Semantic Search, Los Angeles, June 2010.')
    hmm.decode('Query Routing: Finding Ways in the Maze of the Deep Web. G. Kabra, C. Li, and K. C.-C. Chang. In Proceedings of the ICDE International Workshop on Challenges in Web Information Retrieval and Integration (ICDE-WIRI 2005), Tokyo, Japan, April 2005. (14/47=30%).')
    hmm.decode('MetaQuerier over the Deep Web: Shallow Integration across Holistic Sources. K. C.-C. Chang, B. He, and Z. Zhang. In Proceedings of the VLDB Workshop on Information Integration on the Web (VLDB-IIWeb 2004), Toronto, Canada, August 2004. (20/42=48%).')
    hmm.decode('On-the-fly Constraint Mapping across Web Query Interfaces. Z. Zhang, B. He, and K. C.-C. Chang. In Proceedings of the VLDB Workshop on Information Integration on the Web (VLDB-IIWeb 2004), Toronto, Canada, August 2004. (20/42=48%).')
    hmm.decode('Mining Complex Matchings across Web Query Interfaces. B. He, K. C.-C. Chang, and J. Han. In Proceedings of the 9th ACM SIGMOD Workshop on Research Issues on Data Mining and Knowledge Discovery (SIGMOD-DMKD 2004), pages 3-10, Paris, France, June 2004. (8/34=24%).')
    hmm.decode('Clustering Structured Web Sources: A Schema-Based, Model-Differentiation Approach.. B. He, T. Tao, and K. C.-C. Chang. In EDBT Workshops (EDBT-ClustWeb 2004), pages 536-546, Crete, Greece, March 2004. (9/15=60%).')
    hmm.decode('Database Research at the University of Illinois at Urbana-Champaign. M. Winslett, K. C.-C. Chang, A. Doan, J. Han, C. Zhai, and Y. Zhou. SIGMOD Record, 31(3):97-102, September 2002.')
    hmm.decode('The Stanford InfoBus and Its Service Layers: Augmenting the Internet with Higher-Level Information Management Protocols. M. Roscheisen, M. Baldonado, K. C.-C. Chang, L. Gravano, S. Ketchpel, and A. Paepcke. In Digital Libraries in Computer Science: The MeDoc Approach, Lecture Notes in Computer Science No. 1392, pages 213-230. 1998. PS')
    hmm.decode('Deep-Web Search. K. C.-C. Chang. In Encyclopedia of Database Systems, pages 784-788. Springer US, 2009.')
    hmm.decode('PowerBookmarks: An Advanced Web Bookmark Database System and its Information Sharing and Management. W.-S. Li, Y.-L. Wu, C. Bufi, K. C.-C. Chang, D. Agrawal, and Y. Hara. In Information Organization and Databases, chapter 26. Kluwer Academic Publishers, 2000.')
    hmm.decode('The UIUC Web Integration Repository. K. C.-C. Chang, B. He, C. Li, and Z. Zhang. Computer Science Department, University of Illinois at Urbana-Champaign. http://metaquerier.cs.uiuc.edu/repository, 2003.')
    hmm.decode('Query and Data Mapping Across Heterogeneous Information Sources. K. C.-C. Chang. PhD thesis, Stanford Univ., January 2001. PS')    