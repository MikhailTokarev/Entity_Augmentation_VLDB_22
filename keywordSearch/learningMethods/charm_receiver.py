import math
import os
import pickle

from datatype.features import FeatureConstructor
from datatype.whooshengine import KeywordSearch

RECORD_DIR_PATH = "/data/datatype/idfStats/"

class ReceiverCharmKeyword(object):
	"""docstring for ReceiverCharm"""
	def __init__(self, data, receiverData, dataSource, datarecord, dbIndexSuffix):
		super(ReceiverCharmKeyword, self).__init__()
		self.invertedIndex = dict()
		self.numOfFeatures = dict()
		self._translationWeights = dict()
		self.dataPath = dataSource
		self.returnedTuples = list()
		self.receivedSignals = list()

		self.receiver = KeywordSearch(receiverData, dataSource, dbIndexSuffix)

		self.setupStrategy(data, datarecord)

	def save_obj(self, obj, name):
		with open(name + '.pkl', 'wb') as f:
			pickle.dump(obj, f)

	def load_obj(self, name):
		with open(name + '.pkl', 'rb') as f:
			return pickle.load(f)

	def setupStrategy(self, data, datarecord):
		print('Setting Up Strategy')
		featureConst = FeatureConstructor()
		listOfTuples = data.getValues()
		lengthOfFeatures = list()
		countTermInDoc = dict()
		self.idf = dict()

		if not os.path.exists(RECORD_DIR_PATH+self.dataPath+datarecord+"/"):
			os.makedirs(RECORD_DIR_PATH+self.dataPath+datarecord+"/")
			for record in listOfTuples:

				intentFeatures = featureConst.getFeaturesOfSingleTuple(record, data.getHeader(), 1)
				tupleID = record[0]

				if tupleID not in self.numOfFeatures:
					self.numOfFeatures[tupleID] = list()
				for intentFeature in intentFeatures:
					if intentFeature not in self.invertedIndex:
						self.invertedIndex[intentFeature] = list()
					if tupleID not in self.invertedIndex[intentFeature]:
						self.invertedIndex[intentFeature].append(tupleID)
					if intentFeature not in self.numOfFeatures[tupleID]:
						self.numOfFeatures[tupleID].append(intentFeature)
					if intentFeature not in countTermInDoc:
						countTermInDoc[intentFeature] = 1
					else:
						countTermInDoc[intentFeature] += 1
				lengthOfFeatures.append(len(self.numOfFeatures[tupleID]))

	        # setting up external words statistics
			numDocuments = len(listOfTuples)
			for term in countTermInDoc:
				self.idf[term] = math.log(numDocuments / float(countTermInDoc[term]))
			maxIDF = max(list(self.idf.values()))
			for term in self.idf:
				self.idf[term] = self.idf[term]/maxIDF

			print('saving stats')
			self.save_obj(self.idf, RECORD_DIR_PATH+self.dataPath+datarecord+"/idf")
			self.save_obj(self.invertedIndex, RECORD_DIR_PATH+self.dataPath+datarecord+"/invertedIndex")
			self.save_obj(self.numOfFeatures, RECORD_DIR_PATH+self.dataPath+datarecord+"/numOfFeatures")
		else:
			print("loading data...")
			self.idf = self.load_obj(RECORD_DIR_PATH+self.dataPath+datarecord+"/idf")
			self.invertedIndex = self.load_obj(RECORD_DIR_PATH+self.dataPath+datarecord+"/invertedIndex")
			self.numOfFeatures = self.load_obj(RECORD_DIR_PATH+self.dataPath+datarecord+"/numOfFeatures")

	def returnTuples(self, signalsReceived, numberToReturn):

		self.receivedSignals = signalsReceived
		signalsReceived = signalsReceived[0]

		returnedIDs, contentPartTime, bm25Scores = self.receiver.search(signalsReceived, 500)

		if len(returnedIDs) == 0:
			return []

		return returnedIDs[:numberToReturn]