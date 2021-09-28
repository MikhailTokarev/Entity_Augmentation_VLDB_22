from datatype.datastore import Data
from learningMethods.charm_sender import SenderCharm


class Sender(object):
    """docstring for Sender"""

    def __init__(self):
        super(Sender, self).__init__()
        self.data = None

    def initialize(self, data_type, file_to_store, SELECTION_MODE):
        self.strategy = SenderCharm(self.data, data_type, file_to_store, SELECTION_MODE)
        if SELECTION_MODE == 're':
            self.strategy.constructSelectionStrategy()

    def setData(self, filename, file_type):
        self.data = Data(filename, file_type)
        return len(self.data.getHeader())

    def pickTupleToJoin(self):
        self.currentTuples = self.strategy.pickTuplesToJoin()
        return self.currentTuples

    def pickSignals(self, intents=None, how_many=2):
        self.signals = self.strategy.selectSignals(intents, how_many)

        return self.signals

    def update_model_shared(self, terms_sent, tupleID, terms_in_matching_tuple, reinforcement):
        return self.strategy.update_model_shared(terms_sent, tupleID, terms_in_matching_tuple, reinforcement)

    def modelIsFitted(self):
        return self.strategy.model.is_fitted()

    def getWeights(self):
        return self.strategy.getWeights()

    def processExternalData(self, data):
        self.strategy.processExternalData(data)

    def addTerms(self, intent, receiverRecord):
        return self.strategy.addTerms(intent, receiverRecord)

    def reinforceSelectionStrategy(self, intent, signalsToReinforce, score):
        self.strategy.reinforceSelectionStrategy(intent, signalsToReinforce, score)