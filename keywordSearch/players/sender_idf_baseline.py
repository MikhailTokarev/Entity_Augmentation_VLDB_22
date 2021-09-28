from datatype.datastore import Data
from learningMethods.charm_sender import SenderCharm


class SenderIDFBaseline(object):
    """docstring for Sender"""

    def __init__(self):
        super(SenderIDFBaseline, self).__init__()
        self.data = None

    def initialize(self, data_type, file_to_store, SELECTION_MODE):
        self.strategy = SenderCharm(self.data, data_type, file_to_store,
                                    SELECTION_MODE, IDF_BASELINE=True)

    def setData(self, filename, file_type):
        self.data = Data(filename, file_type)
        return len(self.data.getHeader())

    def pickTupleToJoin(self):
        self.currentTuples = self.strategy.pickTuplesToJoin()
        return self.currentTuples

    def pickSignals(self, intents=None, how_many=2, signal_per_attribute=True):
        self.signals = sorted([(splitSignal, self.strategy.idf[splitSignal.keyword])
                     for signal in self.strategy.signalIndex[intents[0]]
                     for splitSignal in signal.splitByAttribute()], key=(lambda x: x[1]), reverse=True)[:how_many]

        return self.signals

    def update_model_shared(self, terms_sent, tupleID, terms_in_matching_tuple, reinforcement):
        return

    def modelIsFitted(self):
        return False

    def reinforceSelectionStrategy(self, intent, signalsToReinforce, score):
        return
