from hmm import HMM
from feature import FeatureGenerator
from tokens import Tokens
import math, sys
from utils import get_binary_vector, log_err, log_1p, log_sum, to_label
from string import punctuation
from datetime import datetime
from training_set_generator import get_training_samples, get_training_samples_author
import numpy as np
import operator
import ast
from language_model import FeatureEntityList, LanguageModel
from boosting_feature import BoostingFeatureGenerator


class Retrainer(object):
    def __init__(self, raw_segments, observation_sequences, label_sequences):
        super(Retrainer, self).__init__()
        self.raw_segments = raw_segments
        self.observation_sequences = observation_sequences
        self.label_sequences = label_sequences
        self.hmm_new = None
        self.feature_entity_list = FeatureEntityList()
        self.lm = LanguageModel()
        self.boosting_feature_generator = BoostingFeatureGenerator()

        self.DOMINANT_RATIO = 0.85  # dominant label ratio: set empirically

        self.retrain_with_boosting_features()
    
    def retrain(self):
        self.hmm_new = HMM('retrainer', 6)
        self.hmm_new.train(self.observation_sequences, self.label_sequences, useLaplaceRule=False)  #important to set laplace to be no
    
    # With new features
    def retrain_with_boosting_features(self):
        # Build language model
        for raw_segment, label_sequence in zip(self.raw_segments, self.label_sequences):
            for token, label in zip(Tokens(raw_segment).tokens, label_sequence):
                self.lm.add(token, label)
        self.lm.prettify()
        self.token_BGM = self.lm.prettify_model
        self.pattern_BGM = None

        # Retrain
        self.hmm_new = HMM('retrainer', 6)
        partial_features = []
        for raw_segment in self.raw_segments:
            partial_features.append(BoostingFeatureGenerator(raw_segment, self.token_BGM, self.pattern_BGM).features)
        self.hmm_new.train(partial_features, self.label_sequences, useLaplaceRule=False)
        self.observation_sequences = partial_features


    def run(self):
        i = 0
        self.new_labels = []
        for raw_segment, label_sequence in zip(self.raw_segments, self.label_sequences):
            new_labels = self.hmm_new.decode(raw_segment)[1]
            self.new_labels.append(new_labels)
            tokens = Tokens(raw_segment).tokens
            feature_vectors = FeatureGenerator(raw_segment).features
            print i, ':  ', raw_segment
            for token, old_label, new_label, feature_vector in zip(tokens, label_sequence, new_labels, feature_vectors):
                print to_label(old_label), '\t', to_label(new_label), '\t', token
                self.feature_entity_list.add_entity(feature_vector, old_label, token)   #???? Old label first
            print '\n'
            i+=1

    def find_pattern(self):        
        self.hmm_new.feature_entity_list.print_all_entity()


    # Find the first tokens at VN boundaries
    def find_venue_boundary_tokens(self):
        recorder = {}
        for raw_segment, observation_sequence, label_sequence in zip(self.raw_segments, self.observation_sequences, self.label_sequences):
            first_target_label_flag = True
            tokens = Tokens(raw_segment).tokens
            for token, feature_vector, label in zip(tokens, observation_sequence, label_sequence):
                # First meet a VN label
                if label == 4 and first_target_label_flag:
                    key = token.lower()
                    if not key.islower():
                        continue
                    if recorder.has_key(key):
                        recorder[key] += 1
                    else:
                        recorder[key] = 1
                    first_target_label_flag = False

                elif (first_target_label_flag is False) and label in [0,1,3]:
                    first_target_label_flag = True

        for k,v in recorder.iteritems():
            print k, '\t', v
        return recorder


    # Learn the general order of structure of publications before moving forward
    def find_majority_structure(self):
        first_bit_counter = {'0': 0, '3': 0, '4':0, '5':0}
        overall_pattern_counter = {}
        for label_sequence in self.label_sequences:
            label = label_sequence[0]
            if label == 2:
                continue
            elif label == 5:
                continue
            elif label in [0,1]:
                first_bit_counter['0'] += 1
            else:
                first_bit_counter[str(label)] += 1

            pattern = []
            for label in label_sequence:
                if label in [2,5]:
                    continue
                elif label in [0,1]:
                    if 0 in pattern:
                        continue
                    else:
                        pattern.append(0)
                elif label == 3:
                    if 3 in pattern:
                        continue
                    else:
                        pattern.append(3)
                elif label == 4:
                    if 4 in pattern:
                        continue
                    else:
                        pattern.append(4)
            key = str(pattern)
            if overall_pattern_counter.has_key(key):
                overall_pattern_counter[key] += 1
            else:
                overall_pattern_counter[key] = 1

        # Inducing the structure
        sorted_firstbit_counter = sorted(first_bit_counter.iteritems(), key=operator.itemgetter(1), reverse=True)
        sorted_pattern_counter = sorted(overall_pattern_counter.iteritems(), key=operator.itemgetter(1), reverse=True)
        print '===========================================', sorted_pattern_counter
        return int(sorted_firstbit_counter[0][0]), ast.literal_eval(sorted_pattern_counter[0][0]), (float(sorted_pattern_counter[0][1]))/len(self.label_sequences)

    def run_with_boosting_features(self):
        i = 0
        self.new_labels = []
        self.combined_labels = []

        for raw_segment, label_sequence in zip(self.raw_segments, self.label_sequences):
            feature_vectors, new_labels = self.hmm_new.decode(raw_segment, True, True, self.token_BGM, self.pattern_BGM)
            self.new_labels.append(new_labels)
            tokens = Tokens(raw_segment).tokens
            print i, ':  ', raw_segment

            # Combination step: 
            tmp_combined_labels = []    # the decided combined labels so far
            for token, old_label, new_label, feature_vector in zip(tokens, label_sequence, new_labels, feature_vectors):

                # Combine old and new labels to come out a combined label, and deciding...
                combined_label = -1
                
                if old_label == new_label:
                    combined_label = new_label
                    tmp_combined_labels.append(new_label)
                
                # Combine compatible labels: FN and LN
                elif old_label in [0,1] and new_label in [0,1]:
                    combined_label = old_label
                    tmp_combined_labels.append(new_label)
                
                # Combine labels that are not compatible
                else:   
                    tmp_feature_entity = self.hmm_new.feature_entity_list.lookup(feature_vector)    # Get the Background knowledge provided the feature vector: the language feature model
                    sorted_label_distribution = sorted(tmp_feature_entity.label_distribution.iteritems(), key=operator.itemgetter(1), reverse=True)
                    total_label_occurence = float(sum(tmp[1] for tmp in sorted_label_distribution))

                    

                    # ============================================================================================
                    # ============================================================================================
                    # ???? Experimenting: removing the low prob label distribution; FAILURE; ARCHIVED HERE AND DEPRECATED 
                    # sorted_label_distribution = []
                    # sum_prob = 0.0
                    # for pair in tmp_sorted_label_distribution:
                    #     sorted_label_distribution.append(pair)
                    #     sum_prob += pair[1]
                    #     if sum_prob/total_label_occurence >= 0.90:
                    #         break
                    # ============================================================================================
                    # ============================================================================================



                    # Dominant label case: Iterate from the highest label stats according to this feature vector:
                    for label_frequency in sorted_label_distribution:
                        if int(label_frequency[0]) in [old_label, new_label] and (label_frequency[1]/total_label_occurence)>=self.DOMINANT_RATIO:
                            print 'Dominant labels'
                            # Check for constraint:
                            tmp_label_to_check = int(label_frequency[0])
                            
                            # Find last occurence position of this label
                            if tmp_label_to_check not in [0,1]:
                                last_occurence = ''.join([str(c) for c in tmp_combined_labels]).rfind(str(tmp_label_to_check))
                            elif tmp_label_to_check in [0,1]:
                                last_occurence_0 = ''.join([str(c) for c in tmp_combined_labels]).rfind('0')
                                last_occurence_1 = ''.join([str(c) for c in tmp_combined_labels]).rfind('1')
                                last_occurence = max(last_occurence_0, last_occurence_1)

                            # Checking constraints by simplifying what we did in viterbi
                            if last_occurence == -1 or last_occurence == (len(tmp_combined_labels)-1):  # Never occurred, or last occurence is the last label
                                # When we are deciding the first label
                                if len(tmp_combined_labels) == 0:
                                    first_bit = self.find_majority_structure()[0]
                                    if first_bit == 0 and tmp_label_to_check not in [0,1]:
                                        continue
                                    if first_bit == 3 and tmp_label_to_check != 3:
                                        continue

                                # VN CANNOT FOLLOW TI W/O DL constraint
                                if tmp_label_to_check == 4 and tmp_combined_labels[-1] == 3:
                                    continue
                            elif tmp_label_to_check in [0,1]:
                                flag = False
                                for j in range(last_occurence, len(tmp_combined_labels)):
                                    if tmp_combined_labels[j] not in [0,1,2]:
                                        flag = True
                                        break
                                if flag:
                                    continue
                            elif tmp_label_to_check == 3:
                                continue
                            elif tmp_label_to_check == 4:
                                if tmp_combined_labels[-1] == 3:    #????
                                    continue

                            combined_label = tmp_label_to_check
                            tmp_combined_labels.append(tmp_label_to_check)
                            break
                    
                    # No dominance case OR Dominance-fail-due-to-constraint case: Find relatively if the label with higher possibility follow the constraint of publication order
                    if combined_label == -1:
                        # Iterate from the highest label stats according to this feature vector:

                        for label_frequency in sorted_label_distribution:
                            breakout_flag = False
                            #Test against constraints
                            # 1. DL separate labels principle
                            # 2. AU-TI-VN Order 
                            if int(label_frequency[0]) in [old_label, new_label]:
                                tmp_label_to_check = int(label_frequency[0])
                                
                                # find structure of the order, and find what have appeared, and so predict what to be appear next
                                structure_overview = []     #will record the order in big sense: 0,3,4/4,0,3
                                for tmp_combined_label in tmp_combined_labels:
                                    if tmp_combined_label in [2,5]:
                                        continue                                            
                                    elif tmp_combined_label in [0,1]:
                                        if 0 in structure_overview:
                                            continue
                                        else:
                                            structure_overview.append(0)
                                    elif tmp_combined_label == 3:
                                        if 3 in structure_overview:
                                            continue
                                        else:
                                            structure_overview.append(3)
                                    elif tmp_combined_label == 4:
                                        if 4 in structure_overview:
                                            continue
                                        else:
                                            structure_overview.append(4)
                                # Based on the structure overview, find what should appear next
                                appear_next = []
                                if structure_overview == [0]:
                                    appear_next = [0,1,3,2,5]
                                elif structure_overview == [3]:
                                    appear_next = [3,0,1,2,5]
                                elif structure_overview == [0,3]:
                                    appear_next = [3,4,2,5]
                                elif structure_overview == [3,0]:
                                    appear_next = [0,1,4,2,5]
                                elif structure_overview == [0,3,4]:
                                    appear_next = [4,2,5]
                                elif structure_overview == [3,0,4]:
                                    appear_next = [4,2,5]
                                else:   #weird case
                                    print 'Weird structure! Weird case!'
                                    if tmp_feature_entity.label_distribution[str(old_label)] > tmp_feature_entity.label_distribution[str(new_label)]:
                                        tmp_label_to_check_list = [old_label, new_label]
                                    else:
                                        tmp_label_to_check_list = [new_label, old_label]
                                    # Apply constraints here too
                                    for tmp_label_to_check in tmp_label_to_check_list:
                                        if tmp_label_to_check not in [0,1]:
                                            last_occurence = ''.join([str(c) for c in tmp_combined_labels]).rfind(str(tmp_label_to_check))
                                        elif tmp_label_to_check in [0,1]:
                                            last_occurence_0 = ''.join([str(c) for c in tmp_combined_labels]).rfind('0')
                                            last_occurence_1 = ''.join([str(c) for c in tmp_combined_labels]).rfind('1')
                                            last_occurence = max(last_occurence_0, last_occurence_1)

                                        # Checking constraints by simplifying what we did in viterbi
                                        if last_occurence == -1 or last_occurence == (len(tmp_combined_labels)-1):
                                            # When we are deciding the first label
                                            if len(tmp_combined_labels) == 0:
                                                first_bit = self.find_majority_structure()[0]
                                                if first_bit == 0 and tmp_label_to_check not in [0,1]:
                                                    continue
                                                if first_bit == 3 and tmp_label_to_check != 3:
                                                    continue
                                            try:
                                                if tmp_label_to_check == 4 and tmp_combined_labels[-1] == 3:
                                                    continue
                                            except:
                                                continue
                                        elif tmp_label_to_check in [0,1]:
                                            flag = False
                                            for j in range(last_occurence, len(tmp_combined_labels)):
                                                if tmp_combined_labels[j] not in [0,1,2]:
                                                    flag = True
                                                    break
                                            if flag:
                                                continue
                                        elif tmp_label_to_check == 3:
                                            continue
                                        elif tmp_label_to_check == 4:
                                            if tmp_combined_labels[-1] == 3:
                                                continue

                                        combined_label = tmp_label_to_check
                                        tmp_combined_labels.append(combined_label)
                                        breakout_flag = True
                                        break

                                if breakout_flag:
                                    break
                                if tmp_label_to_check in appear_next:
                                    # Then check constraint. find last occurence, DL constraints
                                    # Just need to check DL constraints, no need to verify more on tokens, assume token verification is done in the first iteration
                                    if tmp_label_to_check not in [0,1]:
                                        last_occurence = ''.join([str(c) for c in tmp_combined_labels]).rfind(str(tmp_label_to_check))
                                    elif tmp_label_to_check in [0,1]:
                                        last_occurence_0 = ''.join([str(c) for c in tmp_combined_labels]).rfind('0')
                                        last_occurence_1 = ''.join([str(c) for c in tmp_combined_labels]).rfind('1')
                                        last_occurence = max(last_occurence_0, last_occurence_1)

                                    # Checking constraints by simplifying what we did in viterbi
                                    if last_occurence == -1 or last_occurence == (len(tmp_combined_labels)-1):
                                        if tmp_label_to_check == 4 and tmp_combined_labels[-1] == 3: #Hardcode rule [2013/07/23]: For VN, cannot directly follow a TI without DL???? may remove on real effect
                                            continue
                                    elif tmp_label_to_check in [0,1]:
                                        flag = False
                                        for j in range(last_occurence, len(tmp_combined_labels)):
                                            if tmp_combined_labels[j] not in [0,1,2]:
                                                flag = True
                                                break
                                        if flag:
                                            continue

                                    elif tmp_label_to_check == 3:
                                        continue
                                        # flag = False
                                        # for j in range(last_occurence, len(tmp_combined_labels)):
                                        #     if tmp_combined_labels[j] not in [3,2]:
                                        #         flag = True
                                        #         break
                                        # if flag:
                                        #     continue

                                    elif tmp_label_to_check == 4:
                                        if tmp_combined_labels[-1] == 3:    #????
                                            continue

                                    # elif tmp_label_to_check == 2:
                                    # elif tmp_label_to_check == 5:
                                    
                                    # Otherwise, pass
                                    log_err('\t\t' + str(i) + 'Should combine this one')
                                    combined_label = tmp_label_to_check
                                    tmp_combined_labels.append(tmp_label_to_check)
                                    # combined_label = (tmp_label_to_check, sorted_label_distribution)
                                    break
                                    
                                else:
                                    continue

                        # Debug
                        if combined_label == -1:
                            log_err(str(i) + 'problem')
                            combined_label = (appear_next, sorted_label_distribution)
                            tmp_combined_labels.append(-1)


            # Final check the accordance with the major order, ideally, all records under one domain should have the same order... PS very ugly code I admit
            print '==========================tmp_combined_labels', tmp_combined_labels
            majority_order_structure = self.find_majority_structure()[1]
            majority_rate = self.find_majority_structure()[2]
            tmp_combined_labels_length = len(tmp_combined_labels)
            if majority_rate > 0.80 and majority_order_structure == [0,3,4]:
                # p1(phase1): author segments
                for p1 in range(tmp_combined_labels_length):
                    if tmp_combined_labels[p1] in [0,1,2,5]:
                        continue
                    else:
                        break

                # p2(phase2): title segments
                for p2 in range(p1, tmp_combined_labels_length):
                    if tmp_combined_labels[p2] == 3:
                        continue
                    else:
                        break

                #p3(phase3): venue segments
                for p3 in range(p2, tmp_combined_labels_length):
                    if tmp_combined_labels[p3] in [2,5,4]:
                        continue
                    else:
                        break

                # Decision
                if p1 == 0:
                    print 'Houston we got a SERIOUS problem!'
                    log_err('Houston we got a SERIOUS problem!!!!!!!!')

                if p2 == p1:
                    print 'Houston we got a problem!'
                    for sp2 in range(p2, tmp_combined_labels_length):
                        if tmp_combined_labels[sp2] != 2:
                            tmp_combined_labels[sp2] = 3
                        else:
                            break   # should fix common mislabeling at this point now??????????


            # elif majority_rate > 0.80 and majority_order_structure == [3,0,4]:    # ???? not sure if this is normal
            #     # p1(phase1): title segments
            #     for p1 in range(tmp_combined_labels_length):
            #         if tmp_combined_labels[p1] in [3]:
            #             continue
            #         else:
            #             break

            #     # p2(phase2): author segments
            #     for p2 in range(p1, tmp_combined_labels_length):
            #         if tmp_combined_labels[p2] == 3:
            #             continue
            #         else:
            #             break

            #     #p3(phase3): venue segments
            #     for p3 in range(p2, tmp_combined_labels_length):
            #         if tmp_combined_labels[p3] in [2,5,4]:
            #             continue
            #         else:
            #             break

            #     # Decision
            #     if p1 == 0:
            #         print 'Houston we got a SERIOUS problem!'
            #         log_err('Houston we got a SERIOUS problem!!!!!!!!')

            #     if p2 == p1:
            #         print 'Houston we got a problem!'
            #         for sp2 in range(p2, tmp_combined_labels_length):
            #             if tmp_combined_labels[sp2] != 2:
            #                 tmp_combined_labels[sp2] = 3
            #             else:
            #                 break
            for old_label, new_label, tmp_combined_label, token, feature_vector in zip(label_sequence, new_labels, tmp_combined_labels, tokens, feature_vectors):
                print to_label(old_label), '\t', to_label(new_label), '\t', to_label(tmp_combined_label), '\t', token, '\t', feature_vector
            print '\n'
            i+=1