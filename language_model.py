import operator
from feature import STANDARD_PIPELINE, PARTIAL_PIPELINE

class LanguageModel(object):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.model = {}
        self.prettify_model = {}

    def add(self, token, label):
        token = str(token)  # case sensitive
        if self.model.has_key(token):
            self.model[token][str(label)] += 1
        else:
            self.model[token] = {}
            self.model[token]['0'] = 0
            self.model[token]['1'] = 0
            self.model[token]['2'] = 0
            self.model[token]['3'] = 0
            self.model[token]['4'] = 0
            self.model[token]['5'] = 0
            self.model[token][str(label)] = 1

    def prettify(self):
        for k,v in self.model.iteritems():
            sorted_v = sorted(v.iteritems(), key=operator.itemgetter(1), reverse=True)
            self.prettify_model[k] = sorted_v


class FeatureEntity(object):
    def __init__(self, feature_vector):
        super(FeatureEntity, self).__init__()
        self.feature_vector = feature_vector
        self.label_distribution = {}    # Possible labels on this feature vector
        self.token_distribution = {}    # Possible tokens on this feature vector

    def add_label(self, label):
        label = str(label)
        if self.label_distribution.has_key(label):
            self.label_distribution[label] += 1
        else:
            self.label_distribution[label] = 1
        for i in range(6):
            if not self.label_distribution.has_key(str(i)):
                self.label_distribution[str(i)] = 0

    def add_token(self, token):
        token = str(token)
        if self.token_distribution.has_key(token):
            self.token_distribution[token] += 1
        else:
            self.token_distribution[token] = 1

    def print_entity(self):
        sorted_label_distribution = sorted(self.label_distribution.iteritems(), key=operator.itemgetter(1), reverse=True)
        sorted_token_distribution = sorted(self.token_distribution.iteritems(), key=operator.itemgetter(1), reverse=True)
        total_label_occurence = float(sum(tmp[1] for tmp in sorted_label_distribution))
        total_token_occurence = float(sum(tmp[1] for tmp in sorted_token_distribution))
        stat_sorted_label_distribution = [(key, value, value/total_label_occurence) for key, value in sorted_label_distribution]
        stat_sorted_token_distribution = [(key, value, value/total_token_occurence) for key, value in sorted_token_distribution]

        print 'Feature Vector: ', self.feature_vector
        print 'Top labels:'
        for tmp in stat_sorted_label_distribution:
            print tmp
        print 'Top tokens:'
        for tmp in stat_sorted_token_distribution:
            print tmp


# Sparse add features
class FeatureEntityList(object):
    def __init__(self):
        super(FeatureEntityList, self).__init__()
        self.feature_list = []

    def add_entity(self, feature_vector, label, token=None):
        for i in range(len(self.feature_list)):
            if feature_vector == self.feature_list[i].feature_vector:
                self.feature_list[i].add_label(label)
                if token:
                    self.feature_list[i].add_token(token)
                return

        new_fe = FeatureEntity(feature_vector)
        new_fe.add_label(label)
        if token:
            new_fe.add_token(token)
        self.feature_list.append(new_fe)

    def print_all_entity(self):
        for feature_entity in self.feature_list:
            print '\n\n============================================'
            feature_entity.print_entity()

    # Lookup the feature vector is in the set already
    def lookup(self, feature_vector):
        for i in range(len(self.feature_list)):
            if feature_vector == self.feature_list[i].feature_vector:
                return self.feature_list[i]
        return None


        