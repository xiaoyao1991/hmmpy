import sqlite3
import pickle
from classifier import HMMClassifier

conn = sqlite3.connect('/Users/xiaoyaoqian/Desktop/mp/python/hmm2/StandardTest/segmentation_test.db')
cursor = conn.cursor()

# Train models from pickle
observation_sequences_entire = pickle.load(open('training_observation.pkl', 'rb'))
label_sequences_entire = pickle.load(open('training_labels.pkl', 'rb'))
classifier = HMMClassifier()
classifier.HMMentire.train(observation_sequences_entire, label_sequences_entire)
classifier.HMMentire.print_model()


# data_dict: pub_id, record, title, authors, venue, year
def auto_test_one(data_dict):
    author_field, title_field, venue_field, year_field = classifier.decode(data_dict['record'])
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

    qmark = (
                # data_dict['pub_id'], 
                -1,     #???? temp set as is, because the automatic tests didn't extract from db
                data_dict['record'], 
                author_field, 
                title_field, 
                venue_field, 
                str(year_field), 
                data_dict['authors'],
                data_dict['title'],
                data_dict['venue'],
                str(data_dict['year']),
                False
            )
    cursor.execute("INSERT INTO record_record(pub_id,full_record,predicted_author_field,predicted_title_field,predicted_venue_field,predicted_year_field,correct_author_field,correct_title_field,correct_venue_field,correct_year_field,checked) VALUES (?,?,?,?,?,?,?,?,?,?,?)", qmark)
    conn.commit()


def automatic_test_dump():
    test_list = pickle.load(open('autotest.samples', 'rb'))
    i = 0
    for data_dict in test_list:
        print i, '========================================================='
        auto_test_one(data_dict)
        i+=1


if __name__ == '__main__':
    automatic_test_dump()




        
