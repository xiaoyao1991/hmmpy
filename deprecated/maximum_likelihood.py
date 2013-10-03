import math


class MaximumLikelihoodLearning(object):
    """docstring for MaximumLikelihoodLearning"""
    def __init__(self, model):
        super(MaximumLikelihoodLearning, self).__init__()
        self.model = model
        self.useLaplaceRule = True  # default to true

    
    # param: observation <-- list of lists (each list represent a sentence sequence, encoded in integer, should have mapping of integer to feature vectors)
    def train(self, observation_sequences, label_sequences):
        # Setup model information
        num_samples = len(observation_sequences)
        num_states = self.model.num_states
        num_symbols = self.model.num_symbols

        pi = [0.0] * num_states
        transitions = [[0.0] * num_states] * num_states
        emissions = [[0.0] * num_symbols] * num_states

        log_pi = []
        log_transitions = [[0.0] * num_states] * num_states
        log_emissions = [[0.0] * num_symbols] * num_states

        # 1. Counting first state in every label sequence to form pi
        for label_sequence in label_sequences:
            pi[label_sequence[0]] += 1

        # 2. Count all state transitions
        for label_sequence in label_sequences:
            for j in range(1, len(label_sequence)):
                transitions[label_sequence[j-1]][label_sequence[j]] += 1

        # 3. Count emissions for each label
        for i in range(num_samples):
            for j in range(len(observation_sequences[i])):
                emissions[label_sequences[i][j]][observation_sequences[i][j]] += 1

        # 4. Form log probability, by using Laplace correction to avoid zero probabilities
        if self.useLaplaceRule:
            for i in range(len(pi)):
                pi[i] += 1

                for j in range(num_states):
                    transitions[i][j] += 1
                for k in range(num_symbols):
                    emissions[i][k] += 1

        pi_count = sum(pi)
        transition_count = [sum(transition) for transition in transitions]    #????
        emission_count = [sum(emission) for emission in emissions]

        for i in range(len(pi)):
            log_pi[i] = math.log(pi[i] / pi_count)
        for i in range(num_states):
            for j in range(num_states):
                log_transitions[i][j] = math.log(transitions[i][j] / transition_count[i])
        for i in range(num_states):
            for j in range(num_symbols):
                log_emissions[i][j] = math.log(emissions[i][j] / emission_count[i])


        # ???? already good?
        self.model.pi = log_pi
        self.model.transitions = log_transitions
        self.model.emissions = log_emissions

        # 5. Compute log_likelihood
        # log_likelihood = float('-inf')
        # for observation_sequence in observation_sequences:
        #     log_likelihood = MaximumLikelihoodLearning.log_sum(log_likelihood, model.evaluate(observation_sequence))   #????

        # return logLikelihood;



    @staticmethod
    def log_1p(x):
        if x <= -1.0:
            return float('-inf')    # means something is wrong
        if abs(x) > 0.0001:
            return log(1.0 + x)
        return (-0.5 * x + 1.0) * x

    @staticmethod
    def log_sum(ln1, ln2):
        if ln1 == float('-inf'):
            return ln2
        if ln2 == float('-inf'):
            return ln1

        if ln1 > ln2:
            return ln1 + MaximumLikelihoodLearning.log_1p(math.exp(ln2 - ln1))
        return ln2 + MaximumLikelihoodLearning.log_1p(math.exp(ln1 - ln2))

