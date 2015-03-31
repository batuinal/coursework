'''
Created on March 14, 2015
author: Nikita Bhutani
uniquename: nbhutani
'''
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
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

def main(data_dir):
    features, classifier = train()
    test(data_dir, classifier, features)
    
def train():
    global my_tokenizer, mix_label, joke_label
    train_dir = 'kaggle.training/'
    train_files = os.listdir(train_dir)
    inverted_index = defaultdict(dict)
    for train_file in train_files:
        infile = open(train_dir + train_file)
        docId = getDocId(train_file)
        doc_content = infile.read()
        infile.close()
        indexDocument(doc_content, docId, inverted_index)
    features = getFeatures(inverted_index)
    designMatrix = getDesignMatrix(inverted_index, features, train_files)
     
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
    classifier = LogisticRegression()
    classifier.fit(designMatrix, train_labels)
    return features, classifier

def test(data_dir, classifier, features):
    global my_tokenizer, mix_label, joke_label
    test_files = os.listdir(data_dir)
    output = open('test_results_lr.csv', 'w')
    output.write('File,Class\n')
    print('File,Class')
    for test_file in test_files:
        infile = open(data_dir + test_file)
        doc_content = infile.read()
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
        predicted_label = int(classifier.predict(feature)[0])
        predicted_class = 'joke' if predicted_label == joke_label else 'mix'
        row = test_file + ',' + str(predicted_class) 
        print row       
        output.write("%s\n" % row)
    output.close    

def getDocId(doc_file_name):    
    return int(re.sub('^[a-z]+.', '', doc_file_name).lstrip('0'))
 
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
    main(sys.argv[1])