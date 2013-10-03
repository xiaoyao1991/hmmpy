from classifier import HMMClassifier
import pickle

class AutomaticTest(object):
    def __init__(self):
        super(AutomaticTest, self).__init__()

        # Data Preparation
        self.test_list = pickle.load(open('autotest.samples', 'rb'))

        observation_sequences_entire = pickle.load(open('training_observation.pkl', 'rb'))
        label_sequences_entire = pickle.load(open('training_labels.pkl', 'rb'))
        self.classifier = HMMClassifier()
        self.classifier.HMMentire.train(observation_sequences_entire, label_sequences_entire)
        self.classifier.HMMentire.print_model()

    def test_one(self, data_dict):
        author_field, title_field, venue_field, year_field = self.classifier.decode(data_dict['record'])
        print data_dict['record'].encode('ascii', 'ignore')

        print 'Real title:  ', data_dict['title'].encode('ascii', 'ignore')
        print 'Mine title:  ',  title_field.encode('ascii', 'ignore')
        print '' 
        print 'Real authors:  ', data_dict['authors'].encode('ascii', 'ignore')
        print 'Mine authors:  ',  author_field.encode('ascii', 'ignore')
        print '' 
        print 'Real venue:  ', data_dict['venue'].encode('ascii', 'ignore')
        print 'Mine venue:  ',  venue_field.encode('ascii', 'ignore')
        print '' 

        print 'Real year:  ', data_dict['year']
        print 'Mine year:  ',  year_field
        print ''

    def run(self):
        i = 1
        for data_dict in self.test_list:
            print i, '========================================================='
            self.test_one(data_dict)
            i+=1


if __name__ == '__main__':
    AutomaticTest().run()




        
