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

def main(test_dir, train_dir, test_labels, train_labels):
    train_files, y_train = read_data(train_dir, train_labels)
    test_files, y_test = read_data(test_dir, test_labels)

    features, classifier = train(train_files, y_train)
    test(test_files, y_test, classifier, features)
    """
    skf = StratifiedKFold(y, n_folds=5)
    # split data into 5 sets of train/test data
    for train_index, test_index in skf:
        print 'Starting fold'
        # filenames to train on
        f_train = filenames[train_index]
        y_train = y[train_index]

        f_test = filenames[test_index]
        y_test = y[test_index]

        print 'Training fold'
        features, classifier = train(f_train, y_train)
        print 'Testing fold'
        test(f_test, y_test, classifier, features)
    """

def read_data(dirname, label_file):
    filenames = []
    for path, _, files in os.walk(dirname):
        filenames += [os.path.join(path, filename) for filename in files]
    filenames = sorted(filenames)
    labels = numpy.genfromtxt(fname=label_file, skip_header=1, delimiter=',', usecols=(1), converters={1:lambda s: 1 if s == '1' else -1})
    return numpy.array(filenames), labels

def train(documents, labels):
    print 'Training'
    global my_tokenizer
    inverted_index = indexDocuments(documents)
    features = getFeatures(inverted_index)#list(inverted_index.keys())
    design_matrix = getDesignMatrix(inverted_index, features, documents)
    classifier = LogisticRegression()
    classifier.fit(design_matrix, labels)
    return features, classifier

def test(documents, labels, classifier, features):
    print 'Testing'
    global my_tokenizer
    inverted_index = indexDocuments(documents)
    design_matrix = getDesignMatrix(inverted_index, features, documents)
    predicted = classifier.predict(design_matrix)
    f1_score = metrics.f1_score(labels, predicted, average='micro')
    accuracy = metrics.accuracy_score(labels, predicted)
    print accuracy, f1_score
    return accuracy, f1_score

def getDocId(doc_file_name):
    mid = re.sub('^[A-Za-z\-]+', '', os.path.basename(doc_file_name)).lstrip('0')
    return int(re.sub('.html$', '', mid))

def getFeatures(inverted_index):
    features_list = []
    threshold = 10
    for term in inverted_index.keys():
        document_frequency = len(inverted_index[term])
        if document_frequency > threshold:
            features_list.append(term)
    return features_list

def getDesignMatrix(inverted_index, features, documents):
    design_matrix = numpy.zeros((len(documents), len(features)))
    docids = [getDocId(d) for d in documents]
    for termIndex in range(len(features)):
        term = features[termIndex]
        docs = inverted_index[term]
        for doc in docs:
            design_matrix[docids.index(doc)][termIndex] = inverted_index[term][doc]
    return design_matrix

def indexDocuments(documents):
    inverted_index = defaultdict(dict)
    for data_file in documents:
        with open(data_file) as f:
            docId = getDocId(data_file)
            doc_content = f.read()
            indexDocument(doc_content, docId, inverted_index)
    return inverted_index

def indexDocumentForTag(document_content, doc_id, inverted_index):
    global my_tokenizer
    terms = my_tokenizer.getTokensForTag(document_content, 'title')
    vocab_terms = set(terms)
    for term in vocab_terms:
        term_frequency = terms.count(term)
        inverted_index[term][doc_id] = term_frequency
    return inverted_index

#indexes the document, updates dictionary
def indexDocument(document_content, doc_id, inverted_index):
    global my_tokenizer
    metadata = my_tokenizer.getMetaData(document_content)
    terms = my_tokenizer.getTokens(document_content)
    vocab_terms = set(terms)
    for term in vocab_terms:
        term_frequency = terms.count(term)
        inverted_index[term][doc_id] = term_frequency
    for entry in metadata.keys():
        inverted_index[entry][doc_id] = metadata[entry]
    return inverted_index

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
