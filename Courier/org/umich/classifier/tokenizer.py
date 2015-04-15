'''
Created on Feb 9, 2015
author: Nikita Bhutani
uniquename: nbhutani 
'''

import re
import stemmer
import os
import features
import lxml.html
from lxml.html.clean import Cleaner

class Tokenizer(object):
    
    def __init__(self):
        self.stemmer = stemmer.PorterStemmer()
        self.cleaner = Cleaner(style=True)
        self.stopWords = []
        if os.path.exists('stopWords'):
            self.stopWords = [line.strip() for line in open('stopwords')]
        else:
            self.stopWords = features.stopWords
            
    def cleanText(self, text):
        return self.cleaner.clean_html(text)
    
    #percentage of relevant urls in the text
    def getRelevantURLFrequency(self, text):
        urls = re.findall(r'[href|HREF]+=[\'"]?([^\'" >]+)', text)
        urlsCount = len(urls);
        if urlsCount == 0:
            return 0
        relevantRegex = '('+'|'.join(features.relevantURLsPattern)+')'
        filteredUrls = [m.group(1) for m in (re.search(relevantRegex, l) for l in urls) if m]
        return len(filteredUrls) * 1.0 / urlsCount
    
    #percentage of relevant tags in the text
    def getRelevantTagCount(self, text):
        tags = re.findall('<([a-zA-Z][a-zA-Z0-9]*)[^>]*>', text)
        tagsCount = len(tags)
        if tagsCount == 0:
            return 0
        relevantRegex = '('+'|'.join(features.relevantTags)+')'
        relevantTags = re.findall('<' + relevantRegex + ".*?>", text);
        return len(relevantTags) * 1.0 / tagsCount
    
    def getTitle(self, text):
        title = re.findall(r'<title>(.*?)</title>', text.lower())
        return title
    
    def getNumeralCharacterCount(self, text):
        text = self.cleanText(text);
        return len([digit for digit in text if digit.isdigit()])
    
    def getAnchorTagCount(self, text):
        tags = re.findall('<a(?=\s|>)', text)
        return len(tags)

    def getListTagCount(self, text):
        tags = re.findall('<li>', text)
        return len(tags)
    
    def getParagraphTagCount(self, text):
        tags = re.findall('<p>', text)
        return len(tags)
    
    def getHeaderTagCount(self, text):
        tags = re.findall('<h1>|<h2>|<h3>', text)
        return len(tags)
            
#     def getTokensForTag(self, text, tag):
#         pattern = re.compile('(?<=' + tag+ '>).+(?=</' + tag + ')', re.DOTALL) 
#         fragments = pattern.search(text.lower())
#         if fragments:
#             tagContent = fragments.group()  
#             filtered_text = self.removeSGML(tagContent)
#             tokens = self.tokenizeText(filtered_text)
#             filtered_tokens = self.removeStopwords(tokens)
#             stemmed_tokens = self.stemWords(filtered_tokens)
#             relevantTokens = ([x.lower() for x in stemmed_tokens])
#             return relevantTokens
#         return []

    def getTokensForTag(self, text, tag):
        html_element = lxml.html.fromstring(text)
        fragments = html_element.xpath('//' + tag + '//' + text())
        tagContent = ' '.join(fragments)
        pattern = re.compile('(?<=' + tag+ '>).+(?=</' + tag + ')', re.DOTALL) 
        fragments = pattern.search(text.lower())
        filtered_text = self.removeSGML(tagContent)
        tokens = self.tokenizeText(filtered_text)
        filtered_tokens = self.removeStopwords(tokens)
        stemmed_tokens = self.stemWords(filtered_tokens)
        relevantTokens = ([x.lower() for x in stemmed_tokens])
        return relevantTokens
    
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
    
    def getMetaData(self, text):
        metadata = {}
        #metadata[features.docTitle] = self.getTitle(text)
        metadata[features.documentLength] = len(text)
        metadata[features.relevantTagsCount] = self.getRelevantTagCount(text)
        metadata[features.relevantURLsCount] = self.getRelevantURLFrequency(text)
        metadata[features.numericCharacterCount] = self.getNumeralCharacterCount(text)
        metadata[features.anchorCount] = self.getAnchorTagCount(text)
        metadata[features.listCount] = self.getListTagCount(text)
        metadata[features.headerCount] = self.getHeaderTagCount(text)
        metadata[features.paragraphCount] = self.getParagraphTagCount(text)
        return metadata
        
    def getTokens(self, text):
        text = self.cleanText(text)
        filtered_text = self.removeSGML(text)
        tokens = self.tokenizeText(filtered_text)
        filtered_tokens = self.removeStopwords(tokens)
        stemmed_tokens = filtered_tokens
        #stemmed_tokens = self.stemWords(filtered_tokens)
        stemmed_tokens = [x.lower() for x in stemmed_tokens]
        return stemmed_tokens   
        