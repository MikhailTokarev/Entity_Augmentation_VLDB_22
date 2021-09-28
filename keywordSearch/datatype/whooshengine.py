import csv
import os
import shutil

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from whoosh import qparser
from whoosh.fields import *
from whoosh.index import create_in
from whoosh.index import exists_in
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

RECORD_DIR_PATH = "/data/datatype/"

class KeywordSearch(object):
    """docstring for KeywordSearch"""
    def __init__(self, pathName, datasource, fileToSave):
        super(KeywordSearch, self).__init__()
        self.dataPath = pathName
        self.table = dict()
        self.tableAttributes = dict()
        self.datasource = datasource
        self.fileToSave = fileToSave
        self.indexer = self.createIndex()
        self.queryType = qparser.OrGroup

    # Delete instance variables and data prior to pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'searcher' in state:
            del state['searcher']
        if 'reader' in state:
            del state['reader']
        if 'parser' in state:
            del state['parser']
        return state

    def closeSearcher(self):

        # Initialize searcher and parser once in order to reduce overhead of re-initializing
        self.searcher = None
        self.indexer.searcher(closereader=False)
        self.parser = QueryParser("content", schema=self.indexer.schema, group=self.queryType)
        self.parser.remove_plugin_class(qparser.WildcardPlugin)

    def storeData(self, dataPath):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        with open(dataPath, encoding="latin-1") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            self.header = next(spamreader)
            for row in spamreader:
                newTokenResult = list()
                tokenResult = nltk.word_tokenize(' '.join(row[1:]))
                for token in tokenResult:
                    if token not in stop_words:
                        newTokenResult.append(stemmer.stem(token))
                self.table[row[0]] = ' '.join(newTokenResult)

        with open(dataPath, encoding="latin-1") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key in row:
                    if key == 'id':
                        curId = row[key]
                        self.tableAttributes[row[key]] = dict()
                        break

                for key in row:
                    if key != 'id':
                        newTokenResult = list()
                        tokenResult = nltk.word_tokenize(str(row[key]))
                        for token in tokenResult:
                            if token not in stop_words:
                                newTokenResult.append(stemmer.stem(token))
                        self.tableAttributes[curId][key] = ' '.join(newTokenResult)

    def search(self, origQuery, numberToReturn):
        titles = list()
        content = list()
        scores = list()

        # Initialize searcher and parser once in order to reduce overhead of re-initializing
        if not hasattr(self, 'searcher') or self.searcher == None:
            self.searcher = self.indexer.searcher()
            self.parser = QueryParser("content", schema=self.indexer.schema, group=self.queryType)
            self.parser.remove_plugin_class(qparser.WildcardPlugin)

        searchTerm = []

        for tuple in origQuery:
            keyword, attr = tuple[0].split('\', \'')
            keyword = keyword.replace('(\'', '')

            for key in keyword.split('-'):
                searchTerm.append(key)

        searchTerm = ' '.join(searchTerm)

        query = self.parser.parse(searchTerm)
        results = self.searcher.search(query, limit=numberToReturn)

        for line in results:
            titles.append(line['title'])
            content.append(line['content'])
            scores.append(line.score)

        return titles[:numberToReturn], content[:numberToReturn], scores[:numberToReturn]

    def createIndex(self):

        index_path = RECORD_DIR_PATH+self.datasource+self.fileToSave+"/"

        if not os.path.exists(index_path):
            print('Creating directory: ' + str(index_path))
            os.makedirs(index_path)

        if exists_in(str(index_path)):
            print('Found index, loading...')
            indexer = open_dir(str(index_path))
        else:
            print('Creating index...')
            self.storeData(self.dataPath)
            shutil.rmtree(index_path)
            os.makedirs(index_path)
            schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True, vector=True))
            for h in self.header[1:]:
                schema.add(h, TEXT(stored=True, vector=True))
            indexer = create_in(RECORD_DIR_PATH+self.datasource+self.fileToSave+"", schema)

            writer = indexer.writer()

            for key in self.table:
                row = self.table[key]
                title = key
                writer.add_document(title=title, content=row, **self.tableAttributes[key])

            writer.commit(optimize=True)

        return indexer
