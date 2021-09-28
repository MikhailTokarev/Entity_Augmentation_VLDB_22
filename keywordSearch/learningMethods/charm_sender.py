import math
import os
import pickle
import random

import numpy as np
import pandas as pd
from datatype.features import FeatureConstructor
from learningMethods import featurization_utils
from learningMethods import model
from sklearn.exceptions import NotFittedError

RECORD_DIR_PATH = "/data/datatype/idfStats/"

class SenderCharm(object):
    def __init__(self, data, dataType, fileToStore, SELECTION_MODE, IDF_BASELINE=False):
        super(SenderCharm, self).__init__()
        self.fileToStore = fileToStore
        self.dataType = dataType

        self.external_term_binary = dict()
        self.local_term_binary = dict()
        self.first = True
        self.start = -1
        self.end = -1

        self.processData(data)

        # External Feature Variables
        self.idf_external = dict()
        self.signalIndex_external = dict()
        self.idf_external_bucket = [(0, .21), (.21, .27), (.27, .34), (.34, .42), (.42, .52), (.52, .71), (.71, 1)]
        self.tf_external_bucket = [(0, .15), (.15, .2), (.2, .25), (.25, .34), (.34, .5), (.5, 1)]
        self.countKeywordInDocs_external = dict()
        self.total_external_relevant = dict() #total external terms that appear in local match
        self.total_local_relevant = dict() #total local terms that appear in external match
        self.relevant_term_local = dict()
        self.numDocuments_external = 0
        self.maximumTF_external = dict()
        self.matched_tuples = dict()

        self.computedExternalStats = False

        # Only set up the model if we aren't using the baseline
        if not IDF_BASELINE:
            self.lin_epsilon = 0 # Controls exploration
            self.lin_attribute_headers = data.header[1:]  # Drop ID attribute

            # Store configurations
            self.SELECTION_MODE = SELECTION_MODE    # 'e-greedy': Use epsilon greedy method when selecting keywords
                                                    # 're': Use Roth and Erev method when selecting keywords

            # Determine characteristics to use based on dataset
            charaterization_method_name = 'get_characteristics_{}'.format(dataType.replace('-', '_'))
            if hasattr(featurization_utils, charaterization_method_name) and callable(
                    getattr(featurization_utils, charaterization_method_name)):
                self.CHARACTERIZATION_METHOD = getattr(featurization_utils,
                                                       'get_characteristics_{}'.format(self.dataType.replace('-', '_')))
            else:
                print('Must implement {} in order to use term characterizations'.format(charaterization_method_name))

            self.model = model.LinearModel()

    def save_obj_strat(self, obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj_strat(self, name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def save_obj(self, obj, name):
        with open('zipf/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        with open('zipf/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def constructSelectionStrategy(self):
        self.selectionStrategy = dict()
        for intent in self.signalIndex:
            if intent not in self.selectionStrategy:
                self.selectionStrategy[intent] = dict()
            for signal in self.signalIndex[intent]:
                self.selectionStrategy[intent][signal.keyword] = 1

    def processData(self, data):

        featureConst = FeatureConstructor()
        listOfTuples = data.getValues()
        countKeywordInDocs = dict()

        self.idf = dict()
        self.signalIndex = dict()

        print(RECORD_DIR_PATH + self.dataType + self.fileToStore + "_sender/")
        if not os.path.exists(RECORD_DIR_PATH + self.dataType + self.fileToStore + "_sender/"):
            os.makedirs(RECORD_DIR_PATH + self.dataType + self.fileToStore + "_sender/")
            print('Creating data...')

            for record in listOfTuples:
                intent = record[0]

                signals = featureConst.getSignalsOfSingleTuple(record, data.getHeader())
                for signal in signals:

                    if intent not in self.signalIndex:
                        self.signalIndex[intent] = list()
                    self.signalIndex[intent].append(signal)

                    if signal.keyword not in countKeywordInDocs:
                        countKeywordInDocs[signal.keyword] = 1
                    else:
                        countKeywordInDocs[signal.keyword] += 1

            # Calculate final IDF scores (normalized)
            numDocuments = len(listOfTuples)
            for keyword in countKeywordInDocs:
                self.idf[keyword] = math.log(numDocuments / float(countKeywordInDocs[keyword]))
            maxIDF = max(list(self.idf.values()))
            for keyword in self.idf:
                self.idf[keyword] = self.idf[keyword] / maxIDF

            for keyword in self.idf:
                self.local_term_binary[keyword] = True

            print('saving stats')
            self.save_obj_strat(self.idf, RECORD_DIR_PATH + self.dataType + self.fileToStore + "_sender/idf")
            self.save_obj_strat(self.signalIndex,
                                RECORD_DIR_PATH + self.dataType + self.fileToStore + "_sender/signalIndex")
        else:
            print("loading data...")
            self.idf = self.load_obj_strat(RECORD_DIR_PATH + self.dataType + self.fileToStore + "_sender/idf")
            self.signalIndex = self.load_obj_strat(
                RECORD_DIR_PATH + self.dataType + self.fileToStore + "_sender/signalIndex")
            for keyword in self.idf:
                self.local_term_binary[keyword] = True

    def processExternalData(self, data):

        featureConst = FeatureConstructor()

        print('Calculating external stats...')

        for record in data:
            intent = record[0]

            signals = featureConst.getSignalsOfSingleTuple(record, None)
            for signal in signals:

                if intent not in self.signalIndex_external:
                    self.signalIndex_external[intent] = list()
                self.signalIndex_external[intent].append(signal)

                if signal.keyword not in self.countKeywordInDocs_external:
                    self.countKeywordInDocs_external[signal.keyword] = 1
                else:
                    self.countKeywordInDocs_external[signal.keyword] += 1

        # Calculate final IDF scores
        self.numDocuments_external = len(data)

        for keyword in self.countKeywordInDocs_external:
            self.idf_external[keyword] = math.log(self.numDocuments_external / float(self.countKeywordInDocs_external[keyword]))
        maxIDF = max(list(self.idf_external.values()))
        for keyword in self.idf_external:
            self.idf_external[keyword] = self.idf_external[keyword] / maxIDF

        for local_intent in self.matched_tuples:
            for external_intent in self.matched_tuples[local_intent]:
                for signal in self.signalIndex_external[external_intent]:
                    self.signalIndex[local_intent].append(signal)

        for keyword in self.idf_external:
            self.external_term_binary[keyword] = True

        self.maximumTF_external = {key: max([signal.getTermFrequncy() for signal in self.signalIndex_external[key]]) for key in self.signalIndex_external}
        normalizedTFCount = {key: (signal.getTermFrequncy() / self.maximumTF_external[key]) for key in self.maximumTF_external for signal in self.signalIndex_external[key]}
        self.idf_external_bucket =  self._findBucket(7, list(self.idf_external.values()))
        self.tf_external_bucket =   self._findBucket(6, list(normalizedTFCount.values()))
        self.computedExternalStats = True

    def _findBucket(self, size, data):
        buckets = []
        df = pd.DataFrame(data, columns=['value'])

        bins = df.value.value_counts(bins=size).sort_index(ascending=True).keys().tolist()
        prev = 0
        for interval in bins[:-1]:
            buckets.append((prev, interval.right))
            prev = interval.right
        buckets.append((prev, 1.0))
        return buckets

    def addTerms(self, tupleID, externalRecord):
        local_intent = tupleID
        intent = externalRecord[0]

        if local_intent not in self.matched_tuples:
            self.matched_tuples[local_intent] = list()

            signals_local = self.signalIndex[local_intent]

            if local_intent not in self.relevant_term_local:
                self.relevant_term_local[local_intent] = dict()

            for signal in signals_local:
                if signal.keyword in self.local_term_binary and self.local_term_binary[signal.keyword]:
                    if signal.keyword not in self.total_local_relevant:
                        self.total_local_relevant[signal.keyword] = signal.getTermFrequncy()
                    else:
                        self.total_local_relevant[signal.keyword] += signal.getTermFrequncy()
                    self.relevant_term_local[local_intent][signal.keyword] = signal.getTermFrequncy()

        if intent not in self.matched_tuples[local_intent]:
            self.matched_tuples[local_intent].append(intent)

            featureConst = FeatureConstructor()

            if intent in self.signalIndex_external:
                signals_external = self.signalIndex_external[intent]
                signals_to_return = list(signals_external)
            else:
                signals_external = featureConst.getSignalsOfSingleTuple(externalRecord, None)

            unseen = True # Need to add tuple to stats
            if intent in self.signalIndex_external:
                unseen = False

            for signal in signals_external:
                if signal.keyword not in self.total_external_relevant:
                    self.total_external_relevant[signal.keyword] = signal.getTermFrequncy()
                else:
                    self.total_external_relevant[signal.keyword] += signal.getTermFrequncy()

                if self.computedExternalStats and unseen:
                    if signal not in self.signalIndex[local_intent]:
                        self.signalIndex[local_intent].append(signal)

                    self.signalIndex_external[intent] = list()
                    self.signalIndex_external[intent].append(signal)

                    if signal.keyword not in self.countKeywordInDocs_external:
                        self.countKeywordInDocs_external[signal.keyword] = 1
                    else:
                        self.countKeywordInDocs_external[signal.keyword] += 1

            if self.computedExternalStats and unseen:

                # Calculate final IDF scores
                self.numDocuments_external += 1
                for keyword in self.countKeywordInDocs_external:
                    self.idf_external[keyword] = math.log(self.numDocuments_external / float(self.countKeywordInDocs_external[keyword]))
                maxIDF = max(list(self.idf_external.values()))
                for keyword in self.idf_external:
                    self.idf_external[keyword] = self.idf_external[keyword] / maxIDF

                for keyword in self.idf_external:
                    self.external_term_binary[keyword] = True

                self.maximumTF_external = {key: max([signal.getTermFrequncy() for signal in self.signalIndex_external[key]]) for key in self.signalIndex_external}
                normalizedTFCount = {key: (signal.getTermFrequncy() / self.maximumTF_external[key]) for key in self.maximumTF_external for signal in self.signalIndex_external[key]}

                for signal in self.signalIndex_external[intent]:
                    self.signalIndex[local_intent].append(signal)
                signals_to_return = list(self.signalIndex_external[intent])
        return

    def pickTuplesToJoin(self, howMany=1):
        tuples = list()
        while len(tuples) < howMany:
            tuples.append(random.choice(list(self.signalIndex.keys())))

        return tuples

    def pickSingleSignal(self, featureWeights):
        total = sum(featureWeights.values())
        chance = random.uniform(0, 1)
        cumulative = 0
        for signal in featureWeights:
            cumulative += float(featureWeights[signal]) / total
            if cumulative > chance:
                score = featureWeights[signal]
                del featureWeights[signal]
                return signal, score

    def selectSingleTerm(self, featureWeights):
        total = sum(featureWeights.values())
        chance = random.uniform(0, 1)
        cumulative = 0
        for signal in featureWeights:
            cumulative += float(featureWeights[signal]) / total

            if cumulative > chance:
                return signal

    def selectSignals(self, intents, howMany):

        terms_to_send = []

        for intent in intents:
            if self.SELECTION_MODE == 'e-greedy':

                # Get list of keywords ranked by linear model
                exploit_list = self.rankKeywords(intent)

                # Produce list of terms
                new_list = []
                for ranked_keyword in exploit_list:
                    new_list.append((ranked_keyword[0], ranked_keyword[1]))

                exploit_list = new_list
                exploit_list = sorted(exploit_list, key=(lambda x: x[1]), reverse=True)

                terms_to_send = self.selectEpsilonGreedy(exploit_list, howMany)
            elif self.SELECTION_MODE == 're':
                # Select via stochastic weights
                featureWeights = {signal: self.selectionStrategy[intent][signal.keyword] for signal in self.signalIndex[intent]}
                terms_to_send = self.selectStochasticWeights(featureWeights, howMany)
            else:
                print('{} is not a valid SELECTION_MODE'.format(self.SELECTION_MODE))
                exit()

        return terms_to_send

    def rankKeywords(self, intent):

        term_list = [splitSignal for signal in self.signalIndex[intent] for splitSignal in signal.splitByAttribute()]

        featurized_terms = np.array([self.featurize_term(x, intent) for x in term_list])

        # Calculate scores and sort exploitative list
        try: # If partial_fit has not yet been called, decision_function will throw an error
            scores = self.model.predict(featurized_terms)
            exploit_list = zip(term_list, scores)
        except NotFittedError: # Shuffle terms randomly (assume all rankings are 0)
            random.shuffle(term_list)
            exploit_list = [(term, 0) for term in term_list]

        return exploit_list

    def featurize_term(self, term, tupleID):
        return self.CHARACTERIZATION_METHOD(self, term, tupleID)

    def getWeights(self):
        return self.model.get_weights()

    def _update(self, sample_x, sample_y):
        if len(sample_x) > 0:
            return self.model.partial_fit(sample_x, sample_y)

    def _get_update_samples_shared(self, terms_sent, tupleID, terms_in_matching_tuple, reinforcement):
        if terms_in_matching_tuple != None:
            terms_in_matching_tuple = [f.split(',')[0][1:].replace('\'', '') for f in terms_in_matching_tuple]
        else:
            terms_in_matching_tuple = []

        sample_x = []
        sample_y = []

        for i in range(len(terms_sent)):

            x = terms_sent[i]
            sample_x.append(self.featurize_term(x, tupleID))

            # x appears in matched tuple
            if x.keyword in terms_in_matching_tuple:
                sample_y.append(reinforcement)
            else:
                sample_y.append(0)

        return sample_x, sample_y

    def update_model_shared(self, terms_sent, tupleID, terms_in_matching_tuple, reinforcement):
        sample_x, sample_y = self._get_update_samples_shared(terms_sent, tupleID, terms_in_matching_tuple,
                                                             reinforcement)

        return self._update(sample_x, sample_y)

    def reinforceSelectionStrategy(self, intent, signalsToReinforce, score):
        for signal in signalsToReinforce:
            self.selectionStrategy[intent][signal] += score

    def selectEpsilonGreedy(self, exploit_list, howMany):
        terms_to_send = []
        while len(terms_to_send) < howMany and 0 < len(exploit_list):
            chance = random.uniform(0, 1)
            if chance > self.lin_epsilon:  # Exploit
                terms_to_send.append(exploit_list.pop(0))
            else:  # Explore
                random_term = random.choice(exploit_list)
                terms_to_send.append(random_term)
                exploit_list.remove(random_term)

        return terms_to_send

    def selectStochasticWeights(self, featureWeights, howMany):
        terms_to_send = []

        # Keep selecting until we hit howMany and we have keywords left
        while len(terms_to_send) < howMany and 0 < len(featureWeights):
            total = sum(featureWeights.values())
            chance = random.uniform(0, 1)
            cumulative = 0
            for signal in featureWeights:
                cumulative += float(featureWeights[signal]) / total

                if cumulative > chance:
                    terms_to_send.append((signal, featureWeights[signal]))
                    del featureWeights[signal]
                    break

        return terms_to_send