import math, sys
from utils import get_binary_vector, log_err, log_1p, log_sum
from datetime import datetime
from tokens import Tokens
from feature import FeatureGenerator
from training_set_generator import get_training_samples, get_training_samples_author
import numpy as np
import operator

class AbsoluteConvergence(object):
    def __init__(self, max_iter, tolerance):
        super(AbsoluteConvergence, self).__init__()
        self.tolerance = tolerance
        self.max_iter = max_iter;
        self.new_value = None
        self.old_value = None
        self.iter = 0

    def set_new_value(self, v):
        self.old_value = self.new_value
        self.new_value = v
        self.iter += 1

    def has_converged(self):
        if self.old_value is None:
            delta = abs(self.new_value)
        else:
            delta = abs(self.new_value - self.old_value)

        print 'delta: ', delta
        if delta <= self.tolerance:
            return True

        if self.iter >= self.max_iter:
            return True

        if self.new_value is None or abs(self.new_value) == float('inf'):
            log_err('\tProblematic converged!')
            return True
        return False


        

class HMM(object):
    """HMM"""
    def __init__(self, name, states, useLaplaceRule=True):
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
        self.observation_dim = 2    #????preset binary features
        self.encoded_observation_dim = 0    # decimal symbol dimension
        self.feature_symbol_mapping = {}



    def set_pi(self, pi):
        if sum(pi) > 0:
            self.pi = pi
            self.log_pi = []
            for i in range(len(pi)):
                self.log_pi.append(math.log(pi[i]))
        else:
            self.log_pi = pi
            self.pi = []
            for i in range(len(pi)):
                self.pi.append(math.exp(pi[i]))

    # TO-DO: Need to check if overflow/underflow
    def set_transitions(self, transitions):
        shape = (len(transitions), len(transitions))
        if sum(transitions[0]) > 0:
            self.transitions = transitions
            self.log_transitions = np.zeros(shape=shape)
            for i in range(len(transitions)):
                for j in range(len(transitions)):
                    self.log_transitions[i][j] = math.log(transitions[i][j])
        else:
            self.log_transitions = transitions
            self.transitions = np.zeros(shape=shape)
            for i in range(len(transitions)):
                for j in range(len(transitions)):
                    self.transitions[i][j] = math.exp(transitions[i][j])

    def set_emissions(self, emissions):
        shape = (len(emissions), len(emissions[0]))
        if sum(emissions[0]) > 0:
            self.emissions = emissions
            self.log_emissions = np.zeros(shape=shape)
            for i in range(len(emissions)):
                for j in range(len(emissions[0])):
                    self.log_emissions[i][j] = math.log(emissions[i][j])
        else:
            self.log_emissions = emissions
            self.emissions = np.zeros(shape=shape)
            for i in range(len(emissions)):
                for j in range(len(emissions[0])):
                    self.emissions[i][j] = math.exp(emissions[i][j])


    # observations preprocessing
    def building_binary_to_decimal_map(self):
        log_err('\tStart generating binary vectors...')
        possible_features = get_binary_vector(self.encoded_observation_dim, self.feature_dim)

        log_err('\tStart mapping feature vectors to int encoding...')
        for possible_feature in possible_features:
            self.feature_symbol_mapping[str(possible_feature)] = len(self.feature_symbol_mapping)
        log_err('\tFinish mapping...')

    def map_binary_to_decimal(self, observation_sequences):
        # Setup information
        if len(self.feature_symbol_mapping) == 0:
            self.observation_sequences = observation_sequences
            self.training_sample_size = len(observation_sequences)
            self.feature_dim = len(observation_sequences[0][0])
            self.encoded_observation_dim = self.observation_dim ** self.feature_dim
            self.building_binary_to_decimal_map()

        encoded_sequences = []
        for observation_sequence in observation_sequences:
            encoded_vector = []
            for vector in observation_sequence:
                encoded_vector.append(self.feature_symbol_mapping[str(vector)])
            encoded_sequences.append(encoded_vector)

        return encoded_sequences


    # Supervised Learning using the maximum likelihood method
    # observation_sequences: 
    # [
    #     [[1,0,0,1], [0,1,1,1], ....],
    #     [...],
    #     ...
    # ]
    def train(self, observation_sequences, label_sequences):
        # Setup information
        self.observation_sequences = observation_sequences
        self.label_sequences = label_sequences
        self.training_sample_size = len(label_sequences)
        self.feature_dim = len(observation_sequences[0][0])  # feature dimension
        self.encoded_observation_dim = self.observation_dim ** self.feature_dim

        log_err('\tStart generating binary vectors...')
        possible_features = get_binary_vector(self.encoded_observation_dim, self.feature_dim)


        log_err('\tStart mapping feature vectors to int encoding...')
        for possible_feature in possible_features:
            self.feature_symbol_mapping[str(possible_feature)] = len(self.feature_symbol_mapping)
        log_err('\tFinish mapping...')

        pi = np.array([0.0] * self.state_dim)
        transitions = np.array([[0.0] * self.state_dim] * self.state_dim)
        emissions = np.array([[0.0] * self.encoded_observation_dim] * self.state_dim)

        pi_counter = np.array([0.0] * self.state_dim)
        transitions_counter = np.array([[0.0] * self.state_dim] * self.state_dim)
        emissions_counter = np.array([[0.0] * self.encoded_observation_dim] * self.state_dim)

        log_pi = np.array([0.0] * self.state_dim)
        log_transitions = np.array([[0.0] * self.state_dim] * self.state_dim)
        log_emissions = np.array([[0.0] * self.encoded_observation_dim] * self.state_dim)

        log_err('\tStart counting pi...')
        # 1. Counting first state in every label sequence to form pi
        for label_sequence in self.label_sequences:
            pi_counter[label_sequence[0]] += 1.0

        log_err('\tStart counting transitions...')
        # 2. Count all state transitions
        for label_sequence in self.label_sequences:
            for j in range(1, len(label_sequence)):
                transitions_counter[label_sequence[j-1]][label_sequence[j]] += 1.0

        log_err('\tStart counting emissions...')
        # 3. Count emissions for each label
        for i in range(self.training_sample_size):
            for j in range(len(self.observation_sequences[i])):
                symbol = self.feature_symbol_mapping[str(self.observation_sequences[i][j])]    # encode the feature vector into int
                emissions_counter[self.label_sequences[i][j]][symbol] += 1.0

        log_err('\tStart forming log probability...')
        # 4. Form log probability, by using Laplace correction to avoid zero probabilities
        if self.useLaplaceRule:
            for i in range(self.state_dim):
                pi_counter[i] += 1.0

                for j in range(self.state_dim):
                    transitions_counter[i][j] += 1.0
                for k in range(self.encoded_observation_dim):
                    emissions_counter[i][k] += 1.0

        pi_count = sum(pi_counter)
        transition_count = [sum(transition) for transition in transitions_counter]    #????
        emission_count = [sum(emission) for emission in emissions_counter]
        log_err('\tStart computing probability...')
        for i in range(len(pi_counter)):
            pi[i] = pi_counter[i] / pi_count
            log_pi[i] = math.log(pi_counter[i] / pi_count)
        for i in range(self.state_dim):
            for j in range(self.state_dim):
                transitions[i][j] = transitions_counter[i][j] / transition_count[i]
                log_transitions[i][j] = math.log(transitions_counter[i][j] / transition_count[i])
        for i in range(self.state_dim):
            for j in range(self.encoded_observation_dim):
                emissions[i][j] = emissions_counter[i][j] / emission_count[i]
                log_emissions[i][j] = math.log(emissions_counter[i][j] / emission_count[i])

        self.pi = pi
        self.transitions = transitions
        self.emissions = emissions
        self.log_pi = log_pi
        self.log_transitions = log_transitions
        self.log_emissions = log_emissions



    ################## Helper functions for Baum-Welch Algorithm ##############
    ################## DON'T USE! PROBLEMMATIC                   ##############
    # ALPHAi(t) = P(o1o2...ot, X=xt | model) 
    # ALPHAj(t+1) = Sum{i=1, N} [ ALPHAi(t) * A[xi, xj] * B[xj, ot+1] ]
    @deprecated
    def alpha(self, observation):
        # 1. Initilization
        # alpha representation: alpha[t][i]
        T = len(observation)
        alpha = np.zeros(shape=(T, self.state_dim))
        for i in range(self.state_dim):
            alpha[0][i] = self.pi[i] * self.emissions[i][observation[0]]

        # 2. Induction
        for t in range(1, T):
            obs = observation[t]
            for i in range(self.state_dim):
                tmp_sum = 0.0
                for j in range(self.state_dim):
                    tmp_sum += alpha[t-1][j] * self.transitions[j][i]
                alpha[t][i] = tmp_sum * self.emissions[i][obs]

        return alpha

    # Helper, private
    @deprecated
    def __log_alpha(self, observation, ln_alpha):
        # 1. Initilization
        # alpha representation: alpha[t][i]
        T = len(observation)
        for i in range(self.state_dim):
            ln_alpha[0][i] = self.log_pi[i] + self.log_emissions[i][observation[0]]

        # 2. Induction
        for t in range(1, T):
            obs = observation[t]
            for i in range(self.state_dim):
                tmp_sum = float('-inf')
                for j in range(self.state_dim):
                    tmp_sum = log_sum(tmp_sum, ln_alpha[t-1][j] + self.log_transitions[j][i])
                ln_alpha[t][i] = tmp_sum + self.log_emissions[i][obs]

        return ln_alpha

    @deprecated
    def log_alpha(self, observation):
        # 1. Initilization
        T = len(observation)
        ln_alpha = np.zeros(shape=(T, self.state_dim))
        ln_alpha = self.__log_alpha(observation, ln_alpha)

        # 2. Induction
        loglikelihood = float('-inf')
        for i in range(self.state_dim):
            loglikelihood = log_sum(loglikelihood, ln_alpha[T-1][i])
        
        return ln_alpha, loglikelihood

    @deprecated
    def beta(self, observation):
        # 1. Initialization
        T = len(observation)
        beta = np.zeros(shape=(T, self.state_dim))
        for i in range(self.state_dim):
            beta[T-1][i] = 1.0

        # 2. Induction
        for t in range(T-2, -1, -1):
            for i in range(self.state_dim):
                tmp_sum = 0.0
                for j in range(self.state_dim):
                    tmp_sum += self.transitions[i][j] * self.emissions[j][observation[t+1]] * beta[t+1][j]
                beta[t][i] += tmp_sum

        return beta

    @deprecated
    def __log_beta(self, observation, ln_beta):
        # 1. Initilization
        T = len(observation)
        for i in range(self.state_dim):
            ln_beta[T-1][i] = 0

        # 2. Induction
        for t in range(T-2, -1, -1):
            for i in range(self.state_dim):
                tmp_sum = float('-inf')
                for j in range(self.state_dim):
                    tmp_sum = log_sum(tmp_sum, ln_beta[t+1][j] + self.log_transitions[i][j] + self.log_emissions[j][observation[t+1]])
                ln_beta[t][i] += tmp_sum

        return ln_beta

    @deprecated
    def log_beta(self, observation):
        # 1. Initilization
        T = len(observation)
        ln_beta = np.zeros(shape=(T, self.state_dim))
        ln_beta = self.__log_alpha(observation, ln_beta)

        # 2. Induction
        loglikelihood = float('-inf')
        for i in range(self.state_dim):
            loglikelihood = log_sum(loglikelihood, ln_beta[0][i] + self.log_pi[i] + self.log_emissions[i][observation[0]])
        
        return ln_beta, loglikelihood

    @deprecated
    def forward_backward(self, observation):
        ln_alpha, ln_alpha_ll = self.log_alpha(observation)
        ln_beta, ln_beta_ll = self.log_beta(observation)
        return ln_alpha, ln_beta, ln_alpha_ll, ln_beta_ll

    # local_log_ksi = log_ksi[i]
    @deprecated
    def ksi(self, index, observation, ln_alpha, ln_beta, log_ksi):
        T = len(observation)

        for t in range(T-1):
            ln_sum = float('-inf')
            x = observation[t+1]

            for i in range(self.state_dim):
                for j in range(self.state_dim):
                    log_ksi[index][t][i][j] = ln_alpha[t][i] + self.log_transitions[i][j] + self.log_emissions[j][x] + ln_beta[t+1][j]
                    ln_sum = log_sum(ln_sum, log_ksi[index][t][i][j])

            for i in range(self.state_dim):
                for j in range(self.state_dim):
                    log_ksi[index][t][i][j] -= ln_sum

        return log_ksi


    @deprecated
    def baum_welch(self, observation_sequences=None, encoded_sequences=None, max_iter=15, tolerance=0.001):
        # ???? testing only
        if observation_sequences:
            observation_sequences = self.map_binary_to_decimal(observation_sequences)
        if encoded_sequences:
            observation_sequences = encoded_sequences

        convergence = AbsoluteConvergence(max_iter, tolerance)

        # 1. Initialization
        N = len(observation_sequences)
        logN = math.log(N)
        log_ksi = []
        log_gamma = []

        for i in range(N):
            T = len(observation_sequences[i])
            log_ksi.append([])
            log_gamma.append(np.zeros(shape=(T, self.state_dim)))
            for t in range(T):
                log_ksi[i].append(np.zeros(shape=(self.state_dim, self.state_dim)))

        stop = False

        TMax = max([len(obs) for obs in observation_sequences])
        # ln_alpha = np.zeros(shape=(TMax, self.state_dim))
        # ln_beta = np.zeros(shape=(TMax, self.state_dim))

        new_ll = float('-inf')
        old_ll = float('-inf')

        # 2. Iterate until convergence or max iterations is reached
        while not stop:
            # for each sequence in the observation_sequences
            for i in range(N):
                observation = observation_sequences[i]
                T = len(observation)
                tmp_log_gamma = log_gamma[i]
                # w = log_weights[i]

                # 1st step: calculating the forward and backward prob for each HMM state
                ln_alpha, ln_beta, ln_alpha_ll, ln_beta_ll = self.forward_backward(observation)

                # 2nd step: determining the freq of the transition-emission pair values, and dividing it by the prob of the entire string
                # compute the gamma values for next computations
                for t in range(T):
                    ln_sum = float('-inf')
                    for k in range(self.state_dim):
                        tmp_log_gamma[t][k] = ln_alpha[t][k] + ln_beta[t][k]# + w
                        ln_sum = log_sum(ln_sum, tmp_log_gamma[t][k])

                    # Normalize if different from zero
                    if ln_sum != float('-inf'):
                        for k in range(self.state_dim):
                            tmp_log_gamma[t][k] = tmp_log_gamma[t][k] - ln_sum

                # Calculate ksi values for next computations
                log_ksi = self.ksi(i, observation, ln_alpha, ln_beta, log_ksi)

                # Compute loglikelihood for the given sequence 
                new_ll = float('-inf')
                for j in range(self.state_dim):
                    new_ll = log_sum(new_ll, ln_alpha[T-1][j])

            # Average the loglikelihood for all sequences
            new_ll /= N
            convergence.set_new_value(new_ll)

            # Check for convergence
            if not convergence.has_converged():
                print 'not converged!'
                # 3. Continue with the param re-estimation
                old_ll = new_ll
                new_ll = float('-inf')

                # 3.1 Re-estimate of initial state prob
                for i in range(len(self.log_pi)):
                    ln_sum = float('-inf')
                    for k in range(N):
                        ln_sum = log_sum(ln_sum, log_gamma[k][0][i])

                    print '>>>>>>>shoud update pi, ln_sum: ', ln_sum, ',  logN: ', logN, ', i: ', i
                    self.log_pi[i] = ln_sum - logN

                # 3.2 Re-estimate of transition probabilities
                for i in range(self.state_dim):
                    for j in range(self.state_dim):
                        ln_num = float('-inf')
                        ln_den = float('-inf')
                        for k in range(N):
                            T = len(observation_sequences[k])
                            for t in range(T-1):
                                ln_num = log_sum(ln_num, log_ksi[k][t][i][j])
                                ln_den = log_sum(ln_den, log_gamma[k][t][i])
                        
                        print '>>>>>>>shoud update trans'    
                        if ln_num == ln_den:
                            self.log_transitions[i][j] = 0
                        else:
                            self.log_transitions[i][j] = ln_num - ln_den

                # Update the emission prob matrix
                for i in range(self.state_dim):
                    for j in range(self.encoded_observation_dim):
                        ln_num = float('-inf')
                        ln_den = float('-inf')
                        for k in range(N):
                            T = len(observation_sequences[k])
                            gamma_k = log_gamma[k]

                            for t in range(T):
                                if observation_sequences[k][t] == j:
                                    ln_num = log_sum(ln_num, gamma_k[t][i])
                                ln_den = log_sum(ln_den, gamma_k[t][i])
                        print '>>>>>>>shoud update emit'
                        self.log_emissions[i][j] = ln_num - ln_den


            else:
                print 'converged?'
                stop = True

        # Update the non_log params
        for i in range(len(self.log_pi)):
            self.pi[i] = math.exp(self.log_pi[i])
        for i in range(len(self.log_transitions)):
            for j in range(len(self.log_transitions)):
                self.transitions[i][j] = math.exp(self.log_transitions[i][j])
        for i in range(len(self.log_emissions)):
            for j in range(len(self.log_emissions[0])):
                self.emissions[i][j] = math.exp(self.log_emissions[i][j])

        return new_ll


    ####################### End of Baum-Welch from C sharp ####################
    ####################### DON'T USE! PROBLEMMATIC!       ####################


    def print_model(self):
        print 'Pi:\n' , self.pi
        print 'Transition Prb:\n', self.transitions
        print 'Emission Prb:\n', self.emissions
        print 'Log Pi:\n' , self.log_pi            
        print 'Log Transition Prb:\n', self.log_transitions
        print 'Log Emission Prb:\n', self.log_emissions


    # Return the probability of generating given observation sequence pairing with given label sequence by this model
    def evaluate_helper(self, observation_sequence, label_sequence):
        # Sum all the log likelihood
        likelihood = 0.0
        for i in range(len(observation_sequence)):
            symbol = self.feature_symbol_mapping[str(observation_sequence[i])]
            if i == 0:
                state_curr = label_sequence[i]
                likelihood = self.pi[state_curr] * self.emissions[state_curr][symbol]
            else:
                state_prev = label_sequence[i-1]
                state_curr = label_sequence[i]
                likelihood *= self.transitions[state_prev][state_curr] * self.emissions[state_curr][symbol]

        return likelihood

    def evaluate_helper2(self, encoded_sequence, label_sequence):
        # Sum all the log likelihood
        likelihood = 0.0
        for i in range(len(encoded_sequence)):
            symbol = encoded_sequence[i]
            if i == 0:
                state_curr = label_sequence[i]
                likelihood = self.pi[state_curr] * self.emissions[state_curr][symbol]
            else:
                state_prev = label_sequence[i-1]
                state_curr = label_sequence[i]
                likelihood *= self.transitions[state_prev][state_curr] * self.emissions[state_curr][symbol]

        return likelihood

    # Exhaustive method on how to evaluate the P(X | HMM) = Sum(P(X|y) where y belong to Y and Y is all possible hidden state sequences)
    def evaluate_exhaustive(self, raw_segment):
        # 1. Translate token list to feature vectors
        observation_sequence = FeatureGenerator(raw_segment).features

        # 2. Encode the feature vectors into integers
        encoded_sequence = []
        for vector in observation_sequence:
            key = str(vector)
            encoded_symbol = self.feature_symbol_mapping[key]
            encoded_sequence.append(encoded_symbol)

        # print encoded_sequence
        print observation_sequence

        # 3. Get the space of all possible hidden state sequence
        sequence_length = len(observation_sequence)
        likelihood = 0.0
        Y = get_binary_vector(self.observation_dim ** sequence_length, sequence_length)
        for y in Y:
            likelihood += self.evaluate_helper(observation_sequence, y)

        return likelihood


    # Calculate P(observation_sequence | HMM)
    def evaluate(self, raw_segment, log=True, normalize=False):
        # 1. Translate token list to feature vectors
        observation_sequence = FeatureGenerator(raw_segment).features

        # 2. Encode the feature vectors into integers
        encoded_sequence = []
        for vector in observation_sequence:
            key = str(vector)
            encoded_symbol = self.feature_symbol_mapping[key]
            encoded_sequence.append(encoded_symbol)

        # print encoded_sequence
        print observation_sequence

        # 3. Call the forward algorithm
        likelihood, log_likelihood = self.forward(encoded_sequence)

        # 4. Normalize the score by length  ???? not work
        if normalize:
            likelihood = float(len(observation_sequence)) * likelihood
            log_likelihood = math.log(float(len(observation_sequence))) + log_likelihood

        if log:
            return log_likelihood
        return likelihood


    def decode(self, raw_segment):
        # 1. Translate token list to feature vectors
        fg = FeatureGenerator(raw_segment)
        observation_sequence = fg.features
        observation_tokens = fg.tokens

        # 2. Encode the feature vectors into integers
        encoded_sequence = []
        for vector in observation_sequence:
            key = str(vector)
            encoded_symbol = self.feature_symbol_mapping[key]
            encoded_sequence.append(encoded_symbol)

        # 3. Call the viterbi algorithm
        decoded_label_sequence = self.viterbi_with_constraint(encoded_sequence, observation_tokens)
        return observation_sequence, decoded_label_sequence


    def decode_without_constraints(self, raw_segment):
        # 1. Translate token list to feature vectors
        fg = FeatureGenerator(raw_segment)
        observation_sequence = fg.features
        observation_tokens = fg.tokens

        # 2. Encode the feature vectors into integers
        encoded_sequence = []
        for vector in observation_sequence:
            key = str(vector)
            encoded_symbol = self.feature_symbol_mapping[key]
            encoded_sequence.append(encoded_symbol)

        # 3. Call the viterbi algorithm
        decoded_label_sequence = self.viterbi(encoded_sequence)
        return observation_sequence, decoded_label_sequence


    # Performing forward algorithm on non-log probability
    def forward(self, encoded_sequence):
        # 1. Detect if encoded_sequence length is 1     ???? should we discard length = 1?
        # if len(encoded_sequence) == 1:
        #     return 0.0, float('-inf')

        # 2. Initialize accumulator, which will be kept updating along with the dynamic programming, and calculate the first step
        accumulator = np.array([0.0] * self.state_dim)
        for i in range(self.state_dim):
            # Pi(Sunny) * P(Dry|Sunny)
            accumulator[i] = self.pi[i] * self.emissions[i][encoded_sequence[0]]

        # 3. Dynamically induce and update the accumulator
        # P(Damp|Sunny) * [ P(Sunny|Sunny)*Pa + P(Sunny|Cloudy)*Pb + P(Sunny|Rainy)*Pc ]
        # transition matrix: transitions[yesterday_idx][today_idx]
        for i in range(1, len(encoded_sequence)):
            tmp_accumulator = np.array([0.0] * self.state_dim) 
            for j in range(self.state_dim):
                accumulated_prb = 0.0
                for k in range(self.state_dim):
                    accumulated_prb += self.transitions[k][j] * accumulator[k]
                tmp_accumulator[j] = self.emissions[j][encoded_sequence[i]] * accumulated_prb
            accumulator = tmp_accumulator

        # 4. Return the sum of the accumulator
        # print 'Likelihood: ', sum(accumulator)
        return sum(accumulator), math.log(sum(accumulator))


    # Given the observation sequence, find the most likely hidden sequence
    def viterbi(self, encoded_sequence):
        V = [{}]
        path = {}

        # 1. Initialize at t = 0
        for i in range(self.state_dim):
            V[0][str(i)] = self.pi[i] * self.emissions[i][encoded_sequence[0]]
            path[str(i)] = [i]

        # 2. Run viterbi for t > 0
        for t in range(1, len(encoded_sequence)):
            V.append({})
            new_path = {}

            for i in range(self.state_dim):
                (prob, state) = max([(V[t-1][str(j)] * self.transitions[j][i] * self.emissions[i][encoded_sequence[t]], j) for j in range(self.state_dim)])
                V[t][str(i)] = prob
                new_path[str(i)] = path[str(state)] + [i]
            path = new_path

        (prob, state) = max([(V[len(encoded_sequence)-1][str(j)], j) for j in range(self.state_dim)])
        return path[str(state)]


    # Return the best path under some constraints
    def viterbi_with_constraint(self, encoded_sequence, tokens):
        V = [{}]
        path = {}

        # 1. Initialize at t = 0
        for i in range(self.state_dim):
            V[0][str(i)] = self.pi[i] * self.emissions[i][encoded_sequence[0]]
            path[str(i)] = [i]

        # 2. Run viterbi for t > 0
        for t in range(1, len(encoded_sequence)):
            V.append({})
            new_path = {}

            print tokens[:t]
            print sorted(V[-2].iteritems(), key=operator.itemgetter(1), reverse=True)
            print path
            print '\n'
            for i in range(self.state_dim):
                counter = {}
                for j in range(self.state_dim):
                    counter[str(j)] = V[t-1][str(j)] * self.transitions[j][i] * self.emissions[i][encoded_sequence[t]]


                # max[Pa * P(sunny|sunny), Pb * P(sunny|cloudy)...] --> just for sunny, 
                # AND SEE WHICH PRODUCT IS THE LARGEST, THEN DECIDE THAT STATE AS THE PREVIOUS STATE, THUS RECURSIVELY DECIDE ON THE WHOLE SEQUENCE
                max_list = sorted(counter.iteritems(), key=operator.itemgetter(1), reverse=True)
                # print max_list

                # Looping through the max's, from the largest to smallest
                valid_flag = False
                for k in range(len(max_list)):
                    # if k == len(max_list) - 1:   # ???? Essentially means this is the last possible state, and its prob is small
                        # print 'THE LAST ONE!! ', max_list[k]
                    (state, prob) = max_list[k]
                    prev_states = path[str(state)]

                    # Apply constraint:
                    # if the new state is in previous states:
                    # 1. if the last state is delimiter, then check if the new state is the second last state, if not, then not allow
                    # 2. if the last state is not delimiter, then check if the new state is the last state, if not, then not allow
                    # 3. 0,1 states(FN, LN) states need special consideration
                    last_occurence = ''.join([str(c) for c in prev_states]).rfind(str(i))
                    if last_occurence == -1 or last_occurence == (len(prev_states)-1) or i==2:
                        valid_flag = True
                        break

                    # Name position constraints
                    elif i in [0,1]:    #???? Probably need to loosen this constraint, cuz now we see more name patterns
                        # ???? Name Constraint Proposal: As long as the first state in the prev state that is not 2, is in [0,1], then any 0,1 is allowed
                        # if prev_states[-1] == 2 and prev_states[-2] in [0,1]:
                        #     valid_flag = True
                        #     break
                        # elif prev_states[-1] in [0,1]:
                        #     valid_flag = True
                        #     break
                        # else:
                        #     continue
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

                        if sub_valid_flag:
                            valid_flag = True
                            break
                        else:
                            continue

                    elif i == 3:
                        # print 'NOTALLOW 2, \tstate:', state, '\tnewlabel: ', i
                        continue
                    elif i == 5:
                        if not tokens[t].isdigit() or int(tokens[t])<1980 or int(tokens[t])>datetime.now().year:
                            continue
                        else:
                            valid_flag = True
                            break
                    else:
                        valid_flag = True
                        break
                        # prev_states[-1] = prev_states[-1]
                        # if prev_states[-1] == 2 and last_occurence == (len(prev_states)-2):
                        #     valid_flag = True
                        #     # print state
                        #     break
                        # else:
                        #     print 'NOTALLOW 3, \tstate:', state, '\tnewlabel:', i
                        #     continue

                # print 'Max List: ', max_list
                # Update the thang
                if valid_flag:# and (-1 not in path[str(state)]):
                    V[t][str(i)] = prob
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
                else:   #????????????????????????????????????????????????????????????????????????????????
                    # print 'NOTALLOW 3 --> ', i
                    V[t][str(i)] = 0.0   #???? TOO EMPIRICAL
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
            # print 'path: ', path


        ## ???? BUG REPORT
        #  The bug here is due to that til some step in the middle, every thing in the V is all 0.
        #  so that, the order of sorted dictionary will be the same order of key order, which is not correct
        #  0.0 times anything is always still 0.0.
        print 'Total V: ', sorted(V[-1].iteritems(), key=operator.itemgetter(1), reverse=True)
        print 'Total path: ', path
        (prob, state) = max([(V[len(encoded_sequence)-1][str(j)], j) for j in range(self.state_dim)])
        return path[str(state)]


    # TO_DO: For normalizing probability to sequence length
    def z_score(self, x):
        pass



if __name__ == '__main__':
    # observation_sequences, label_sequences = get_training_samples_author('http://scholar.google.com/citations?user=WZ7Pk9QAAAAJ&hl=en')
    # # observation_sequences, label_sequences = get_training_samples_author('http://scholar.google.com/citations?user=x3LTjz0AAAAJ&hl=en')  #small sample for debugging
    # log_err('Training sample retrieved!')
    # hmm = HMM('author', 2)
    # log_err('Start Training...')
    # hmm.train(observation_sequences, label_sequences)
    # hmm.print_model()
    # test1 = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]#, [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
    # test2 = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
        
    # print hmm.evaluate(test1)
    # print hmm.evaluate(test2)



    ############################################################
    # leeds example
    # hmm = HMM('test', 3)
    # hmm.pi = [0.63, 0.17, 0.20]
    # hmm.transitions = [[0.5, 0.25, 0.25],[0.375, 0.125, 0.375],[0.125, 0.675, 0.375]]
    # hmm.emissions = [[0.60, 0.20, 0.15, 0.05], [0.25, 0.25, 0.25, 0.25], [0.05, 0.10, 0.35, 0.50]]

    # test_encoded_sequence = [0, 3, 2,1,0]
    # # print hmm.forward(test_encoded_sequence)
    # print hmm.viterbi(test_encoded_sequence)

    
    ############################################################
    # Testing correctness of baum-welch using dishonest gambler
    hmm = HMM('test', 2)
    hmm.set_pi([0.5, 0.5])

    B = np.array( [ [ 1.0/6,  1.0/6,  1.0/6,  1.0/6,  1.0/6,  1.0/6, ], \
                       [ 1.0/10, 1.0/10, 1.0/10, 1.0/10, 1.0/10, 1.0/2 ] ] )

    A = np.array( [ [ 0.99, 0.01 ], \
                       [ 0.01, 0.99 ] ] )

    hmm.set_transitions(A)
    hmm.set_emissions(B)

    print "\nDishonest Casino Example:\n "
    obs = [[ 0,1,0,5,5 ]]
    
    hmm.baum_welch(encoded_sequences=obs)
    print '>>>>>'
    hmm.print_model()


