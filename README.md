## HMM Based publication segmentation


#### Cora specific
- Maybe using the stanford-ner's tokenizer, which seems much more fancier than mine.


#### Directory Structure
- /data: Contains some dictionaries that contains certain tokens. These files ended in *.lst
- /deprecated: Some .py files that are not used
- /log: Test logs will be stored here
- batch_test.sh: a shell script that calls test.py on the testing samples.
- hmm.py: The HMM model class, and its function implementation(viterbi, training)
- feature.py: The basic feature class. Contains features in both 1st iteration and 2nd iteration
- boosting_feature.py: Extended the basic feature class, and contains additional features that used in 2nd iteration.
- languange_model.py: The background knowledge model used in the Combine stage.
- tokens.py: Tokenize the publications and also apply some preprocessing.
- training_set_generator.py: The class that crawl the Google Scholar, and generate different versions of a single piece of record to enlarge the training set.
- utils.py: Some utility methods.
- classifier.py: The wrapper class that will do the publication segmentation work.
- retrainer: The class that will do the retraining job. This is where the 2nd iteration happens


#### Usage:
Run the test.py to see detail instructions. 

#### Caution:
The training set generating step may take long. And so every time the crawler actually goes to the google scholar to crawl data, the results will be serialized locally, and thus the data can be reused.