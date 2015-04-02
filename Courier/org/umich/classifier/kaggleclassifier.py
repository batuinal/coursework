'''
Created on March 14, 2015
author: Nikita Bhutani
uniquename: nbhutani
'''
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
import sys
import os
import tokenizer
import numpy
import re

my_tokenizer = tokenizer.Tokenizer()
joke_label = 1
mix_label = 0
punctuation_count_feature = 'punctuationcount'
non_punctuation_count_feature = 'nonpunctuationcount'
doc_length_feature = 'doclength'
pronoun_count_feature = 'pronouncount'

def main(test_dir, train_dir, label_file):
    filenames, y = read_data(train_dir, label_file)

    skf = StratifiedKFold(y, n_folds=5)
    # split data into 5 sets of train/test data
    for train_index, test_index in skf:
        # filenames to train on
        f_train = filenames[train_index]
        y_train = y[train_index]

        f_test = filenames[test_index]
        y_test = y[test_index]
        
        features, classifier = train(f_train, y_train)
        test(f_test, y_test, classifier, features)

def read_data(dirname, label_file):
    filenames = []
    for path, _, files in os.walk(dirname):
        filenames += [os.path.join(path, file) for file in files]
    filenames = sorted(filenames)
    labels = numpy.genfromtxt(fname=label_file, skip_header=1, delimiter=',', usecols=(1), converters={1:lambda s: 1 if s == 1 else -1})

    return numpy.array(filenames), labels
    
def train(f_train, y_train):
    global my_tokenizer, mix_label, joke_label
    inverted_index = defaultdict(dict)
    for train_file in f_train:
        with open(train_file) as f:
            docId = getDocId(train_file)
            doc_content = f.read()
            indexDocument(doc_content, docId, inverted_index)
    features = getFeatures(inverted_index)
    designMatrix = getDesignMatrix(inverted_index, features, train_files)

    """
    train_labels = numpy.zeros(len(train_files)) 
    train_label_file = 'solution.train.csv'
    infile = open(train_label_file)
    labels = infile.readlines()
    for i in range(1, len(labels)):
        row = labels[i].split(',')
        doc_id = getDocId(row[0])
        label = joke_label if row[1].strip() == 'joke' else mix_label
        train_labels[doc_id - 1] = label
    infile.close()
    """
    classifier = LogisticRegression()
    classifier.fit(designMatrix, y_train)
    return features, classifier

def test(f_test, y_test, classifier, features):
    global my_tokenizer, mix_label, joke_label
#    output = open('test_results_lr.csv', 'w')
#    output.write('File,Class\n')
#    print('File,Class')
    y_predicted = []
    for test_file in f_test:
        with open(test_file) as f:
            doc_content = f.read()
            text_length, punctuation_count, non_punctuation_count,  = my_tokenizer.getFeatures(doc_content)
            tokens = my_tokenizer.getTokens(doc_content)
            pronoun_count = my_tokenizer.getPronounCount(tokens)
            feature = numpy.zeros(len(features))
            for token in set(tokens):
                if token in features:
                    token_index = features.index(token)
                    term_count = tokens.count(token)
                    feature[token_index] = term_count
            feature[features.index(punctuation_count_feature)] = punctuation_count
            feature[features.index(non_punctuation_count_feature)] = non_punctuation_count
            feature[features.index(doc_length_feature)] = text_length
            feature[features.index(pronoun_count_feature)] = pronoun_count
            y_predicted.append(int(classifier.predict(feature)[0]))

    # calc and print accuracy score of the prediction
    print metrics.accuracy_score(y_test, y_predicted)
#            predicted_class = 'joke' if predicted_label == joke_label else 'mix'
#            row = test_file + ',' + str(predicted_class) 
#            print row       
#            output.write("%s\n" % row)
#    output.close    

def getDocId(doc_file_name):
    mid = re.sub('^[A-Za-z]+.', '', os.path.basename(doc_file_name)).lstrip('0')
    return int(re.sub('.html$', '', mid))
 
def getDesignMatrix(inverted_index, features, documents): 
    design_matrix = numpy.zeros((len(documents), len(features)))
    for termIndex in range(len(features)):
        term = features[termIndex]
        docs = inverted_index[term]
        for doc in docs:
            design_matrix[doc - 1][termIndex] = inverted_index[term][doc] 
    return design_matrix

def getFeatures(inverted_index):
    return list(inverted_index.keys())
        
def indexDocument(document_content, doc_id, inverted_index):  
    global my_tokenizer
    text_length, punctuation_count, non_punctuation_count = my_tokenizer.getFeatures(document_content)
    terms = my_tokenizer.getTokens(document_content)
    pronoun_count = my_tokenizer.getPronounCount(terms)
    vocab_terms = set(terms)
    for term in vocab_terms:
        term_frequency = terms.count(term)
        inverted_index[term][doc_id] = term_frequency
    inverted_index[punctuation_count_feature][doc_id] = punctuation_count
    inverted_index[non_punctuation_count_feature][doc_id] = non_punctuation_count    
    inverted_index[doc_length_feature][doc_id] = text_length
    inverted_index[pronoun_count_feature][doc_id] = pronoun_count
    print pronoun_count
    return inverted_index

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
