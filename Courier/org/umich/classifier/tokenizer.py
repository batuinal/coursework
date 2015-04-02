'''
Created on Feb 9, 2015
author: Nikita Bhutani
uniquename: nbhutani 
'''

import re
import stemmer
import os

class Tokenizer(object):
    
    def __init__(self):
        self.stemmer = stemmer.PorterStemmer()
        self.stopWords = []
        self.punctuation = ['!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~'];
        if os.path.exists('stopWords'):
            self.stopWords = [line.strip() for line in open('stopwords')]
        else:
            self.stopWords = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'few', 'from', 'for', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on', 'or', 'our', 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that', 'this', 'to', 'us', 'was', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you', 'your']
        self.pronouns = ['he', 'she', 'it', 'you', 'yours', 'his', 'hers', 'him', 'her', 'i', 'me', 'we', 'its', 'they', 'them', 'theirs', 'their', 'that', 'which', 'who', 'my'];
    
    def removeSGML(self, text):
        trimmedText = re.sub('<[^>]*>\n*', '', text)
        return trimmedText
    
    def getSentences(self, text):
        endOfLine = re.compile("([\.\?\!\:\;][\'\"]?[\s\n]+[\'\"]?\w)")
        newLine = re.compile("[\n\s]+")
        tokens = endOfLine.split(newLine.sub(" ", text))
        sentences = [""]
        for idx in xrange(len(tokens)):
            if idx % 2:
                sentences[-1] += tokens[idx][0]
                sentences.append(tokens[idx][-1])
            else:
                sentences[-1] += tokens[idx]
        return sentences
    
    def tokenizeText(self, text):
        sentences = self.getSentences(text)
        tokens = []
        for sentence in sentences:
            sentence = re.sub(r'^[\"\']|(``)|([ (\[{<])[\"\']', r' ', sentence)
            sentence = re.sub(r'([:,])([^\d|\\?|/?])', r'  \2', sentence) #replace : and , with whitespace
            sentence = re.sub(r'\.\.(\.)*', r' ', sentence) #replace two or more instances of . with whitespace
            if re.compile(r'[+a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,6}').search(sentence) is None : sentence = re.sub(r'[@]', r' \g<0> ', sentence)
            if re.compile(r'([$])(\d+(?:\.\d{2})?)([$%]*)').search(sentence) is None: sentence = re.sub(r'[$%]', r' ', sentence)
            sentence = re.sub(r'[;#&]', r' \g<0> ', sentence) #breaks urls for now but in general it should be good
            sentence = re.sub(r'(?![A-Z])([^\.])(\.)([\]\)}>"\']*)\s*$', r'\1 \2\3 ', sentence)
            sentence = re.sub(r'[?!]', r' ', sentence)
            sentence = re.sub(r'[\]\[\(\)\{\}\<\>]', r' \g<0> ', sentence)
            sentence = re.sub(r'--', r' -- ', sentence)
            sentence = " " + sentence + " " #add extra space to add padding around cases that couldn't be captured in above
            sentence = re.sub(r'"', " '' ", sentence) #ending quotes
            sentence = re.sub(r'(\S)(\'\')', r'\1 \2 ', sentence) #ending quotes
            sentence = re.sub(r"([^' ])(')([mM]|re|RE) ", r"\1 a\3 ", sentence)
            sentence = re.sub(r"([^' ])(')(ll|LL) ", r"\1 wi\3 ", sentence)
            sentence = re.sub(r"([^' ])(')(ve|VE) ", r"\1 ha\3 ", sentence)
            sentence = re.sub(r"([^' ])(n't|N'T) ", r"\1 not ", sentence) #fails for can't, ain't and won't and similar
            sentence = re.sub(r"([^' ])('[sS]) ", r"\1 \2 ", sentence) #possesive
            sentence = re.sub(r"([sS])(')", r"\1 's", sentence) #plural possessive
            sentence = re.sub(r"([^' ])('[dD]|') ", r"\1 \2 ", sentence)
            sentenceTokens = sentence.split();
            for token in sentenceTokens:
                if not re.match(r'^[\W]+$', token):
                    tokens.append(token.lower())
        return tokens
    
    def removeStopwords(self, tokens):
        filteredTokens = [item for item in tokens if item.lower() not in self.stopWords]
        return filteredTokens

    def stemWords(self, tokens):
        stemmedTokens = [self.stemmer.stem(item, 0, len(item) - 1) for item in tokens]
        return stemmedTokens
    
    def getFeatures(self, text):
        text_length = len(text)
        punctuation_count = 0
        for m in self.punctuation:
            punctuation_count = punctuation_count + text.count(m)
        white_space_count = text.count(' ')
        non_punctuation_count = text_length - punctuation_count - white_space_count
        return text_length, punctuation_count, non_punctuation_count  
        
    def getTokens(self, text):
        filtered_text = self.removeSGML(text)
        tokens = self.tokenizeText(filtered_text)
        #filtered_tokens = self.removeStopwords(tokens)
        #stemmed_tokens = self.stemWords(tokens)
        stemmed_tokens = [x.lower() for x in tokens]
        return stemmed_tokens
    
    def getPronounCount(self, tokens):
        pronoun_count = 0
        for m in self.pronouns:
            pronoun_count = pronoun_count + tokens.count(m)
        return pronoun_count    
        